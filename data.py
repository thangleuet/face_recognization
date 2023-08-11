
from enum import IntEnum, unique
from mio_libs import mio_json

@unique
class Response_status(IntEnum):
    ERROR = 0
    NO_ERROR = 1

@unique
class File_type(IntEnum):
    UNKNOW = 0
    IMAGE = 1
    VIDEO = 2

@unique
class Data_type(IntEnum):
    RAW = 1
    RESULT = 2

@unique
class Gender_type(IntEnum):
    UNKNOW = 2
    MAN = 0
    WOMAN = 1

@unique
class Status(IntEnum):
    DEFAULT = 1
    ANALYZING = 2
    FINISH_ANALYZING = 3
    ERROR_ANALYZING = 4

@unique
class Post_status(IntEnum):
    DEFAULT = 1
    POSTING = 2
    ERRORPOSTING = 3
    SUCCESS = 4

@unique
class StatusCamera(IntEnum):
    INACTIVE = 1
    ACTIVE = 2
    DELETED = 3

@unique
class proc_status(IntEnum):
    OK = 1
    ERR_ANALYZE = 2
    ERR_POST_PROC = 3
    
class Point(mio_json.serializable):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def _custom_dict(self):
        return {
            "IsEmpty": False,
            "X": self.x,
            "Y": self.y
        }
    
class Box(mio_json.serializable):
    def __init__(self, x= 0, y= 0, w= 0, h= 0):
        self.x:float = x
        self.y:float = y
        self.w:float = w
        self.h:float = h

    def _custom_dict(self):
        return {
            "Location": Point(self.x, self.y)._custom_dict(),
            "Size": f"{self.w}, {self.h}",
            "X": self.x,
            "Y": self.y,
            "Width": self.w,
            "Height": self.h,
            "Left": self.x,
            "Top": self.y,
            "Right": self.x + self.w,
            "Bottom": self.y + self.h,
            "IsEmpty": False
        }


class CamInfo():
    cam_id:int = None
    shop_id:int = None
    status:StatusCamera = None
    resolution:str = None #1920,1080
    roi:Box = None #31, 173, 1841, 853
    image_rectangle = None
    real_width_cm:float = None
    real_height_cm:float = None
    enable_grid:int = None
    grid_cell_width_cm:float = None
    grid_cell_height_cm:float  = None
    enable_head_size_filter:int = None
    min_head_size:int = None
    enable_back_head_filter:int = None
    delete_data_setting:int = 0

class CamService():
    camid:int = None
    service_id:list = []
    service_name:list = []
    
class Record():
    request_id:str = None
    cam_id:int = None
    shop_id:int = None
    token:str = None
    data_type:Data_type = None
    callback_url:str = None
    object_name:str = None 
    status:Status = None
    post_status:Post_status = None
    cam_code:str = None

class CameraSetting():
    id:int = None
    status = None
    resolution = None
    roi = None
    image_rectangle = None
    real_width_cm = None
    real_height_cm = None
    enable_grid = None
    grid_cell_width_cm = None
    grid_cell_height_cm = None
    enable_head_size_filter = None
    min_head_size = None
    enable_back_head_filter = None
    delete_data_setting = None

class CameraService():
    cam_id = None
    service_id = None
    service_code = None

class Part(mio_json.serializable):
    id:int = None
    name:str = None
    score:float = None

    def _custom_dict(self):
        return {
            "ID": self.id,
            "Name": self.name,
            "score": self.score
        }

class Age(mio_json.serializable):
    def __init__(self, ages=0):
        self.ages = ages
        
    def _custom_dict(self):
        return {
            "MinAge": 0 if self.ages is None  else self.ages,
            "MaxAge": 0 if self.ages is None  else self.ages
        }


@unique
class Headbox_type(IntEnum):
    HEAD_BOX_OK = 1 
    HEAD_BOX_TOO_SMALL = 2
    HEAD_BOX_BACK = 3
    
class Person(mio_json.serializable):
    def __init__(self):
        self.id: int = None
        self.name: str = None
        self.age:Age = None
        self.age_score: float = 0.
        self.gender:Gender_type = Gender_type.UNKNOW
        self.gender_score:float = 0.
        self.parts:list[Part] = None
        self.bodybox:Box = None
        self.facebox:Box = None
        self.headbox:Box = None
        self.headbox_type:Headbox_type = Headbox_type.HEAD_BOX_OK
        self.center:Point = None
        self.center_score:float = 0.0
        self.foot_center:Point = None
        self.foot_center_score:float = 0.0

    def _custom_dict(self):
        return {
            "ID": self.gender if self.gender is not None else -1,
            "Name": self.name,
            "Age": self.age._custom_dict() if self.age != None else Age()._custom_dict(),
            "AgeScore": self.age_score,
            "Gender": self.gender if self.gender is not None else Gender_type.UNKNOW,
            "GenderScore": self.gender_score if self.gender_score is not None else 0.0,
            "Parts": self.parts,
            "BodyBox": self.bodybox._custom_dict() if self.bodybox != None else Box()._custom_dict(),
            "FaceBox": self.facebox._custom_dict() if self.facebox != None else Box()._custom_dict(),
            "HeadBox": self.headbox._custom_dict() if self.headbox != None else Box()._custom_dict(),
            "HeadBox_Type": self.headbox_type,
            "Center": self.center._custom_dict() if self.center!= None else Point()._custom_dict(),
            "CenterScore": self.center_score if self.center_score is not None else 0.0,
            "FootCenter": self.foot_center._custom_dict() if self.foot_center!= None else Point()._custom_dict(),
            "FootCenterScore": self.foot_center_score if self.foot_center_score is not None else 0.0,
        }
        
class AiCamOutput(mio_json.serializable):
    def __init__(self):
        self.status:proc_status = proc_status.OK
        self.record:Record = None
        self.cam_id:int = None
        self.frame_id:int = 0	
        self.total_frames:int = None		
        self.timestampsec:int = None		 
        self.frame = None		           
        self.total_peoples:int = None	   
        self.process_timesec:int = None	 
        self.list_person: list[Person] = None
        self.file_path:str = None
        self.roi:Box = None
    
    def _custom_dict(self):
        return {
                "CamID": self.cam_id,
                "FrameID": self.frame_id,
                # "TotalFrames": self.total_frames,
                "TimestampSec": self.timestampsec,
                # "Frame": None,
                "TotalPeoples": self.total_peoples,
                "ProcessTimeSec": self.process_timesec,
                "ROI": self.roi,
                "Persons": self.list_person,
                "ResultImagePath": self.file_path
            }

if __name__ == "__main__":
    print(Box().to_json())
    # sample = AiCamOutput()
    # sample.status = proc_status.OK
    # sample.record = None
    # sample.cam_id = 18	  
    # sample.frame_id = 0			
    # sample.total_frames = 1800	
    # sample.timestampsec = 500		 
    # sample.frame = None		           
    # sample.total_peoples = 10	   
    # sample.process_timesec = 1300 
    # sample.list_person = None
    # sample.file_path = "6/b09bedc9-15b0-4e03-a8ef-dc99d62fd039/2022/03/21/images_result/output03_21_2022_10_19_30.jpg"
    # sample.roi = Box(0,0,800,600)
    
    # print(sample)
    # print("----------------------------------------------------------------------------------------------------------------------------")
    # print(sample.to_json())
    # print("----------------------------------------------------------------------------------------------------------------------------")
    # print(AiCamOutput.convert_to_json(sample))
    # print("----------------------------------------------------------------------------------------------------------------------------")
    # print(AiCamOutput.convert_to_json([sample, sample]))
    # print("----------------------------------------------------------------------------------------------------------------------------")
    
    # sample.record = Record()
    # sample.record.request_id = "0-4e03-a8ef-dc99d62fd039"
    # sample.record.cam_id = 18
    # sample.record.shop_id = 6
    # sample.record.token = "93312998-d705-4bf8-a147-e51bccb41234"
    # sample.record.data_type = File_type.IMAGE
    # sample.record.callback_url = "https://aicam.miosystem.com:3681/api/AiCam/PostResult"
    # sample.record.object_name = "6/b09bedc9-15b0-4e03-a8ef-dc99d62fd039/2022/03/21/images/03_21_2022_10_19_30.jpg" 
    # sample.record.status = Status.FINISH_ANALYZING
    # sample.record.post_status = Post_status.SUCCESS
    # sample.record.cam_code = "6a28c046-7f58-44d6-b2bf-eaa544a9fb9c"
    
    # p = Person()
    # p.id = 10
    # p.name = "dungtv"
    # p.age = 30
    # p.age_score = 0.92
    # p.gender = Gender_type.MAN
    # p.gender_score = 0.99
    # p.parts = None
    # p.bodybox = Box(0,0,10,20)
    # p.facebox = Box(0,0,30,30)
    # p.headbox = Box(10,20,30,40)
    # p.headbox_type = Headbox_type.HEAD_BOX_OK
    # p.center = Point(5,10)
    # p.center_score = 0.7
    # p.foot_center = Point(60,20)
    # p.foot_center_score = 0.8
        
    # sample.list_person = [p, p]
    
    # print(sample.to_json())
    
    
    
    