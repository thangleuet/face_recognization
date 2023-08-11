from pyclbr import Function
from statistics import mode
import tensorflow as tf 
import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import time
import numpy as np
import os
from numba import cuda
import math

#
# Copyright 2022 by Vmio System JSC
# All rights reserved.
# Utility functions for measuring model performance
#

# References:
# https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f



# Setup environment-------------------------------------------------------------------------------

DEVICE_GPU = "cuda"
DEVICE_TPU = "tpu"
DEVICE_CPU = "cpu"

# Set this before use
__env = "LOCAL" # GG_COLAB --> For Google Colab
__device_type = DEVICE_GPU # Set this if you are using local device

def get_device()->str:
  if __env == "GG_COLAB":
    if int(os.environ["COLAB_GPU"]) > 0:
      return DEVICE_GPU
    if "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
      return DEVICE_TPU
    return DEVICE_CPU
  return __device_type


# Sample code for EfficientNetB0-----------------------------------------------------------------
input_size_model = 100

def load_model() -> Model:
  base_model = kr.applications.EfficientNetB0(
        include_top=False,
        input_shape=(input_size_model, input_size_model, 3),
        pooling="avg")

  features = base_model.output
  pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
  model = Model(inputs=base_model.input, outputs=[pred_gender])
  return model

def take_input(req_num:int)->list:
  inputs = []
  img = np.zeros((input_size_model,input_size_model,3), np.uint8)
  img = np.array(img).reshape(-1, input_size_model, input_size_model, 3)
  img = tf.convert_to_tensor(img)
  for _ in range(req_num):
    inputs.append(img)
  return inputs

@tf.function
def serve(model, input):
  return model(input, training=False)

def serve_batch(model, inputs, batch_num:int):
  return model.predict(np.vstack(inputs),
                       batch_size=batch_num,
                       verbose=0,
                       steps=None,
                       callbacks=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=True)
  
def predict(model, input):
    return serve(model, input)

def predict_batch(model, inputs, batch_num:int = 1):
  return serve_batch(model, inputs, batch_num)


# Measure inference time--------------------------------------------------------------------------------------
def measure_inference_time(load_model_func, take_input_func, predict_func, warmup_repetitions = 10,  repetitions = 100) -> tuple:
  if warmup_repetitions > repetitions:
    print("warmup_repetitions must be less than repetitions")
    return

  # Check device
  device_type = get_device()
  print(f"Testing on {device_type}")
  if device_type == DEVICE_TPU:
    print("device is not supported")
    return

  # Load model
  print("Loading model...")
  model = load_model_func()

  # Take inputs
  print(f"Taking {repetitions} inputs...")
  inputs = take_input_func(repetitions)

  #warm-up
  print(f"Doing warm-up in {warmup_repetitions} iterations...")
  for i in range(warmup_repetitions):
    _ = predict_func(model, inputs[i])

  # Measure performance
  print(f"Measuring performace in {repetitions} iterations")
  timings=np.zeros((repetitions,1))

  if device_type ==DEVICE_GPU:
    # model = cuda.to_device(model) ???
    starter, ender = cuda.event(timing=True), cuda.event(timing=True)
    for i in range(repetitions):
      starter.record()
      _ = predict_func(model, inputs[i])
      ender.record()
      ender.synchronize() # waiting for gpu sync
      timings[i] = starter.elapsed_time(ender)
  else:
    for i in range(repetitions):
        start_time = time.time()
        _ = predict_func(model, inputs[i])
        timings[i] = (time.time() - start_time)*1000
  
  # Sumary
  mean = np.sum(timings) / repetitions
  std = np.std(timings)
  med = np.median(timings)
  return mean, std, med

# Calculate an optimized batch size---------------------------------------------------------------------
# Can estimate the largest batch size using:
# Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
def estimate_batch_size(load_model_func:Function, take_input_func:Function, predict_func_batch:Function, min_batch_size:int = 1, max_batch_size:int = 20, batch_size_step:int =1) -> tuple[int, int]:

  # Load model
  print("Loading model...")
  model = load_model_func()

  # Take inputs
  print(f"Taking {max_batch_size} inputs...")
  inputs = take_input_func(max_batch_size)

  def try_batch(min_batch:int, max_batch:int, step_batch:int) ->int:
    last_ok = 0
    for i in range(min_batch, max_batch + 1, step_batch):
      try:
        print(f"trying batch_size = {i}")
        result = predict_func_batch(model, inputs, i)

        # In some cases, the predict_func_batch always return a valid result because framework adjusted batch_size automatically even memory was overed load
        # So you have to check the log to know when we get the memory limit.
        # Something like: "Allocator (GPU_0_bfc) ran out of memory trying to allocate XXXGiB"
        if result is None: 
          break
        last_ok = i
      except Exception as e:
        print(e)
        break
    return last_ok

  # Try batch size
  print(f"Trying to reach the max batch size: {max_batch_size}")
  last_ok = try_batch(min_batch_size, max_batch_size, batch_size_step)

  # Try with step = 1
  if last_ok >= min_batch_size and last_ok < max_batch_size and batch_size_step > 1:
    last_ok = try_batch(last_ok, max_batch_size, 1)
  
  optimized = int(2**int(math.log(last_ok, 2)))
  
  return last_ok, optimized

# Measure throughput with a specified batch size-----------------------------------------------------------------
def measure_throughput(load_model_func, take_input_func, predict_func_batch, batch_num, warmup_repetitions = 10,  repetitions = 100)->float:
  
  throughput = 0.0

  # Check device
  device_type = get_device()
  print(f"Testing on {device_type}")
  if device_type == DEVICE_TPU:
    print("device is not supported")
    return throughput

  # Load model
  print("Loading model...")
  model = load_model_func()

  # Take inputs
  print(f"Taking {batch_num} inputs...")
  inputs = take_input_func(batch_num)

  #warm-up
  print(f"Doing warm-up in {warmup_repetitions} iterations...")
  for i in range(warmup_repetitions):
    _ = predict_func_batch(model, inputs, 1)

  # Measure performance
  print(f"Measuring throughput with batch_num = {batch_num} in {repetitions} iterations")
  total_time = 0

  if device_type ==DEVICE_GPU:
    # model = cuda.to_device(model) ???
    starter, ender = cuda.event(timing=True), cuda.event(timing=True)
    for i in range(repetitions):
      starter.record()
      _ = predict_func_batch(model, inputs, batch_num)
      ender.record()
      cuda.synchronize() # waiting for gpu sync
      total_time += starter.elapsed_time(ender)
  else:
    for i in range(repetitions):
        start_time = time.time()
        _ = predict_func_batch(model, inputs, batch_num)
        total_time += (time.time() - start_time)
  
  # Sumary
  throughput = (repetitions*batch_num)/total_time
  return throughput


if __name__=="__main__":

  # Disable GPU
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  __env = "LOCAL" # GG_COLAB --> For Google Colab
  __device_type = DEVICE_CPU # Set this if you are using local device

  # mean, std, med = measure_inference_time(load_model, take_input, predict, 10, 100)
  # print(f"Inference time: mean = {mean}ms, stdev={std}ms, median={med}ms")

  # print("--------------------------------------------------------")
  # max_batch_size, optimized_batch_size = estimate_batch_size(load_model, take_input, predict_batch, 33, 41, 1)
  # print(f"Max batch size: {max_batch_size}, Optimized = {optimized_batch_size}")

  optimized_batch_size = 10
  print("--------------------------------------------------------")
  throughput = measure_throughput(load_model, take_input, predict_batch, batch_num=optimized_batch_size, warmup_repetitions=10, repetitions= 50)
  print(f"Throughput: {throughput}/s")