import imp
import cv2
from view.init_ui import predict_position
from main import time_loadmodel
import threading
import asyncio


if __name__ == "__main__":
    list_model = []
    _thread_loadmodel = threading.Thread(target=asyncio.run, args=(time_loadmodel(list_model),))
    _thread_loadmodel.start()
    _thread_loadmodel.join()
    model_yolov5 = list_model[0]
    track = list_model[3]
    img = cv2.imread(r"C:\Users\VMIO\Desktop\image\10_family-slide-top01.jpg")
    list_abc = []
    predict_position(img, list_abc, model_yolov5, track)
    # cv2.imshow("demo", img)

    cv2.waitKey(0)