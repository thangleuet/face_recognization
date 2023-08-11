from distutils.log import set_threshold
import math
from pathlib import Path
import os
import pathlib
import cv2
import numpy as np
import sys
import random

# Project modules
sys.path.append(str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()))

from model.logger import miolog
from mio_libs.mio_ultils import read_class_names, tranfer_image, xyxy2xywh
from omegaconf import OmegaConf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import applications
from tensorflow import keras

# Disable GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_model(model_name, img_size, image_channels):
    try:
        base_model = getattr(applications, model_name)(
            include_top=False,
            input_shape=(img_size, img_size, image_channels),
            pooling="avg"
        )
        features = base_model.output
        pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
        pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
        model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
        return model
    except Exception as e:
        miolog.exception(e,'[AGE_GENDER_RECOGNITE][get_model]')

def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default

def load_model_age_gender(path):
    """Func load model age gender

    Args:
        path (str): Path file model

    Returns:
        object  : model age gender
    """
    model = None
    if os.path.exists(path) == False:
        miolog.error('Func (load_model_age_gender): model age gender does not exist')
        return model
    try:
        model_name, input_size_model = Path(path).stem.split("_")[:2]
        input_size_model = safe_cast(input_size_model, int, 0)

        cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={input_size_model}"])
        model = get_model(model_name, input_size_model, 3)
        model.load_weights(path)
    except Exception as e:
        miolog.exception(e,"[AGE_GENDER_RECOGNITE][load_model_age_gender]")
    return model

def load_model_backhead(path):
    """Func load model back head

    Args:
        path (str): Path file model

    Returns:
        object: model check backhead
    """
    model_backhead = None
    input_size_model_backhead = 100
    if os.path.exists(path) == False:
        miolog.error('Func (load_model_backhead): model age gender does not exist')
        return model_backhead
    try:
        # model_name, input_size_model_backhead = Path(path).stem.split("_")[:2]
        # input_size_model_backhead = safe_cast(input_size_model_backhead, int, 0)
        model_backhead = keras.models.load_model(path, compile=True)
        print("model_backhead", model_backhead)
    except Exception as e:
        miolog.exception(e,"[AGE_GENDER_RECOGNITE][load_model_backhead]")

    return model_backhead


MAN = 0,
WMAN = 1
def _is_fronthead(crop_img, model_backhead, input_size=100, score=0.5, margin=0):
    """Func check fronhead

    Args:
        crop_img (array): image
        model_backhead (objetc): model backhead
        input_size (int, optional): size of model Defaults to 100.
        score (float, optional): threshodl backhead. Defaults to 0.5.
        margin (int, optional): . Defaults to 0.

    Returns:
        bool: True/False
    """
    try:
        # LOGGER.debug("Start predict backhead")
        crop_img = cv2.resize(crop_img, (int(input_size), int(input_size)))

        crop_img = np.array(crop_img).reshape(-1, int(input_size), int(input_size), 3)
        crop_img = tf.convert_to_tensor(crop_img)
        results = model_backhead.predict(crop_img)
        predicted_fronthead = results[0]#.numpy()


        if predicted_fronthead[0] > score:
            backhead_label = True
        else:
            backhead_label = False

        return backhead_label

    except Exception as e:
        miolog.exception(e,"[AGE_GENDER_RECOGNITE][_is_fronthead]")
        return True

@tf.function
def serve(x, model_age_gender):
    return model_age_gender(x, training=False)

def predict_age_gender_EfficientNetB3(img, model_age_gender):
    # global model_EfficientNetB3, input_size_model
    input_size_model = 100
    if model_age_gender is None:
        print("Model is None")
        return None, None, None

    print("Sart predict age/gender")
    img = cv2.resize(img, (int(input_size_model), int(input_size_model)))
    img = np.array(img).reshape(-1, int(input_size_model), int(input_size_model), 3)
    img = tf.convert_to_tensor(img)
    results = serve(img, model_age_gender)  # model_EfficientNetB3.predict(img)

    predicted_genders = results[0].numpy()
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].numpy().dot(ages).flatten()
    age = "{}".format(int(predicted_ages[0]))

    if predicted_genders[0][0] > 0.5:
        score_gender = "{:.2f}".format(predicted_genders[0][0])
        gender = "man"
    else:
        score_gender = "{:.2f}".format(predicted_genders[0][1])
        gender = "woman"

    print("End predict age/gender")
    return age, gender, score_gender

def predict(image, box, margin, model_age_gender):
    try:
        image_h, image_w, _ = image.shape
        x1, y1, w, h = box
        x2 = w + x1
        y2 = h + y1

        # Expand box by margin
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), image_w - 1)
        yw2 = min(int(y2 + margin * h), image_h - 1)

        # Crop image 
        print(f'Image shape: {image.shape}, ROI: {x1},{y1},{x2},{y2}')
        crop_img = image[yw1:yw2 + 1, xw1:xw2 + 1]

        # Predict
        gender_score = random.uniform(0.85, 1.0)
        age = None,
        gender = None
        if crop_img is not None:
            age, gender, gender_score = predict_age_gender_EfficientNetB3(crop_img, model_age_gender)
        return age, gender, safe_cast(gender_score, float, None)

    except Exception as e:
        miolog.exception(e,"[AGE_GENDER_RECOGNITE][predict]")
    return None, None, None

def predict_best(image, box, margin, gender_threshold, model_age_gender):
    # Predict 
    age, gender, gender_score = predict(image, box, margin, model_age_gender)
    gender_score = 0.0 if gender_score is None else gender_score
    print("Predict result: ", age, gender, gender_score)

    # Verify score
    if gender_score < gender_threshold:
        print(f"Score {gender_score} < threshold {gender_threshold} --> start verify")
        margins = [0, 0.1, -0.1]  # , 0.2, -0.2, 0.3, -0.3]
        for i in range(0, len(margins)):
            if math.isclose(margin, margins[i], rel_tol=1e-9):
                print(f"Margin {margins[i]} already tested --> delete")
                del margins[i]
                break
        print(f"Margins for test: {margins}")

        for m in margins:
            print(f"Testing margin = {m}")
            test_age, test_gender, test_gender_score = predict(image, box, m, model_age_gender)
            print("Test result: ", test_age, test_gender, test_gender_score)

            if test_gender_score > gender_score:
                age = test_age
                gender = test_gender
                gender_score = test_gender_score

            if gender_score > 0.9:
                break
            print("-------------------")

    print("Best result:", age, gender, gender_score)
    return age, gender, gender_score


def draw_box_age_gender(image, list_object):
    """Func draw box age gender

    Args:
        image (image): image
        list_object (list[list]): [[xw1, yw1, xw2, yw2, age, gender, gender_score]]

    Returns:
        _type_: _description_
    """
    for i in range(len(list_object)):
        xw1, yw1, xw2, yw2, age, gender, gender_score = list_object[i][0], list_object[i][1], list_object[i][2], list_object[i][3], list_object[i][4], list_object[i][5], list_object[i][6]
        col = (255, 255, 255)
        if age == None and gender == None:
            cv2.rectangle(image,(xw1, yw1), (xw2, yw2), col, 1)
            continue
        if gender == 'M':
            col = (0, 0, 255)
            col_text = (255,255,255)
        if gender == 'W':
            col = (255,0, 0)
            col_text = (7, 147, 235)
        bbox_mess = f'{gender}({age})'
        # if show_label:
        cv2.rectangle(image,(xw1, yw1), (xw2, yw2), col, 1)
        w,h = cv2.getTextSize(bbox_mess,cv2.FONT_HERSHEY_SIMPLEX,  0.5, 1)[0]
        try:
            image = tranfer_image(image, max(0,xw1-3) ,max(0,yw2+3), xw1+w+3, yw2+h+8)
        except Exception as e:
            miolog.exception(e,"[AGE_GENDER_RECOGNITE][draw_box_age_gender]")
        cv2.putText(image, bbox_mess, (xw1,yw2+h+3),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_text, 1, cv2.LINE_AA)

    return image
def predict_age_gender(image, list_head, model_age_gender, model_back_head=None, input_size=100, margin=0, min_size_threshold=-1):
    """Func age gender

    Args:
        image (image): image inputs
        list_head (list[list]]): [[x1,y1,x2,y2]]
        model_age_gender (objec): model
        model_back_head (object, optional): model. Defaults to None.
        input_size (int, optional): size model. Defaults to 100.
        margin (int, optional): . Defaults to 0.
        min_size_threshold (int, optional): min size detect age gender. Defaults to -1.

    Returns:
        output(list): [[list_head[i][0], list_head[i][1], list_head[i][2], list_head[i][3], age, gender, gender_score]]
    """
    list_head_age_gender = []
    if model_age_gender is None:
        miolog.error("Model age gender is None")
        return list_head_age_gender

    s_threshold = min_size_threshold ** 2
    for i in range(len(list_head)):

        age = None
        gender = None
        gender_score = 0.0
        xw1, yw1, xw2, yw2 = None, None, None, None
        try:

            xw1, yw1, xw2, yw2 =  int(list_head[i][0]), int(list_head[i][1]), int(list_head[i][0]) + int(list_head[i][2]),  int(list_head[i][1]) + int(list_head[i][3])
            # Crop image 
            # LOGGER.debug(f'Image shape: {image.shape}, ROI: {xw1},{yw1},{xw2},{yw2}')
            crop_img = image[yw1:yw2 + 1, xw1:xw2 + 1]

            # check size
            if (yw2 - yw1) * (xw2 - xw1) < s_threshold:
                list_head_age_gender.append([xw1, yw1, xw2, yw2, age, gender, gender_score])
                continue

            # check back head
            if model_back_head != None:
                is_fronthead = _is_fronthead(crop_img, model_back_head)
                if is_fronthead == False:
                    list_head_age_gender.append([xw1, yw1, xw2, yw2, age, gender, gender_score])
                    continue

            # predict age/gender
            crop_img = cv2.resize(crop_img, (input_size, input_size), cv2.INTER_AREA)
            results = model_age_gender.predict(np.array(crop_img).reshape(-1, 100, 100, 3))
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            age = int(predicted_ages[0])
            isman = predicted_genders[0][0] > 0.5
            if isman:
                gender = 'M'
                gender_score = "{:.2f}".format(predicted_genders[0][0])
            else:
                gender =  'W'
                gender_score = "{:.2f}".format(predicted_genders[0][1])
        except Exception as e:
            miolog.exception(e,"[AGE_GENDER_RECOGNITE][predict_age_gender]")
        finally:
            list_head_age_gender.append([xw1, yw1, xw2, yw2, age, gender, gender_score])
    return list_head_age_gender

def predict_age_gender_best(image, list_head, model_age_gender, model_back_head=None, input_size=100, margin=0, min_size_threshold=-1):
    list_head_age_gender = []
    if model_age_gender is None:
        miolog.error("Model age gender is None")
        return list_head_age_gender

    s_threshold = min_size_threshold ** 2 if min_size_threshold is not None else 0
    for i in range(len(list_head)):

        age = None
        gender = None
        gender_score = random.uniform(0.85, 1.0)
        xw1, yw1, xw2, yw2 = None, None, None, None
        try:

            xw1, yw1, xw2, yw2 =  int(list_head[i][0]), int(list_head[i][1]), int(list_head[i][0]) + int(list_head[i][2]),  int(list_head[i][1]) + int(list_head[i][3])
            # Crop image 
            # LOGGER.debug(f'Image shape: {image.shape}, ROI: {xw1},{yw1},{xw2},{yw2}')
            crop_img = image[yw1:yw2 + 1, xw1:xw2 + 1]

            # check size
            if (yw2 - yw1) * (xw2 - xw1) < s_threshold:
                list_head_age_gender.append([xw1, yw1, xw2-xw1, yw2-yw1, age, gender, gender_score])
                continue

            # check back head
            if model_back_head != None:
                is_fronthead = _is_fronthead(crop_img, model_back_head)
                if is_fronthead == False:
                    list_head_age_gender.append([xw1, yw1, xw2-xw1, yw2-yw1, age, gender, gender_score])
                    continue

            # predict age/gender
            # crop_img = cv2.resize(crop_img, (input_size, input_size), cv2.INTER_AREA)
            # results = model_age_gender.predict(np.array(crop_img).reshape(-1, 100, 100, 3))
            # predicted_genders = results[0]
            # ages = np.arange(0, 101).reshape(101, 1)
            # predicted_ages = results[1].dot(ages).flatten()
            # age = int(predicted_ages[0])
            # isman = predicted_genders[0][0] > 0.5
            # if isman:
            #     gender = 'M'
            #     gender_score = "{:.2f}".format(predicted_genders[0][0])
            # else:
            #     gender =  'W'
            #     gender_score = "{:.2f}".format(predicted_genders[0][1])

            MARGIN = 0.1
            GENDER_TH = 0.8
            age, gender, gender_score  = predict_best(image, ( xw1, yw1, xw2-xw1, yw2-yw1), MARGIN, GENDER_TH, model_age_gender)
            age = safe_cast(age, int, None)
            gender = None if gender is None else ("M" if gender == "man" else "W")
            list_head_age_gender.append([xw1, yw1, xw2-xw1, yw2-yw1, age, gender, gender_score])
        except Exception as e:
            list_head_age_gender.append([xw1, yw1, xw2-xw1, yw2-yw1, age, gender, gender_score])
            miolog.exception(e,"[AGE_GENDER_RECOGNITE][predict_age_gender_best]")
        # finally:
        #     list_head_age_gender.append([xw1, yw1, xw2-xw1, yw2-yw1, age, gender, gender_score])
    return list_head_age_gender
