import json
import threading
from time import time

from controller.face_recog.facefaiss import FaceRecognition
## ==> SPLASH SCREEN
from view.init_ui import init_component
import asyncio

# ==> GLOBALS
counter = 0

async def time_loadmodel(list_model):

    start_time_facerecog = time()
    model_path = r"model\models\face_recog\arcface_encodings_labels_vmiostaff.obj"
    face_recog = FaceRecognition(model_path,threshold=1)
    end_time_facerecog = time() - start_time_facerecog
    time_loadmodel = {"face_recognition": str(round(end_time_facerecog))}

    list_model.append(face_recog)

    with open("time_loadmodel.txt", "w") as outfile:
        json.dump(time_loadmodel, outfile)


if __name__ == "__main__":

    list_model = []
    _thread_loadmodel = threading.Thread(target=asyncio.run, args=(time_loadmodel(list_model),))
    _thread_loadmodel.start()
    # splash_p = Process(target=init_splash)
    # splash_p.start()
    # splash_p.join()
    _thread_loadmodel.join()

    init_component(list_model)
