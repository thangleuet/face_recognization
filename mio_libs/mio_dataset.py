from __future__ import annotations
from ast import List
from numbers import Number
from pyclbr import Function
import datetime
import json
import os
import cv2
from numpy import number

#
# Copyright 2022 by Vmio System JSC
# All rights reserved.
# Utility functions for creating and converting annotation format
#

def obj_dict(obj):

    """ 
    Args:
        obj (): ogject

    Returns:
        dict: dictionary verison of object
    """    

    return obj.__dict__

def to_linux_path(path:str) ->str:
    return os.path.normpath(path).replace("\\","/")

def get_all_files(dirName:str, exts: list[str]) -> list[str]:
    all_files = []
    try:
        for (dirpath, dirnames, filenames) in os.walk(dirName):
            for file in filenames:
                if any(file.endswith(x)  for x in exts):
                    all_files.append(os.path.join(dirpath, file))
    except Exception as e:
        print(e)
    finally:
        return all_files

class box:
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def from_dict(**entries) ->box:
        obj = box()
        obj.__dict__.update(entries)
        return obj
    
    def __repr__(self):
        return json.dumps(self, default=obj_dict)

class object:
    def __init__(self, box:box = box(), class_names:list[str] = [""]):
        self.box = box
        self.class_names = class_names

    def from_dict(**entries) -> object:
        obj = object()
        obj.__dict__.update(entries)
        return obj
    
    def __repr__(self):
        return json.dumps(self, default=obj_dict)

class image_data():
    def __init__(self, relative_path:str = "", objects:list[object] = []):
        self.relative_path = relative_path
        self.objects = objects
    
    def from_dict(**entries) -> image_data:
        obj = image_data()
        obj.__dict__.update(entries)
        return obj

    def __repr__(self):
        return json.dumps(self, default=obj_dict)

class image_dataset():
    def __init__(self, arr = []):
        self.arr = arr

    def from_dict(**entries) -> image_dataset:
        obj = image_dataset()
        obj.__dict__.update(entries)
        return obj

    def serialize(self, txt_path:str, mode:str='a+t') -> bool:
        try:
            with open(txt_path, mode) as the_file:
                for d in self.arr:
                    try:
                        jsonStr = json.dumps(d, default=obj_dict)
                        the_file.write(jsonStr + "\n")
                    except Exception as e:
                        print(e)
            return True
        except Exception as e:
            print(e)
        
        return False
    
    def deserialize(json_path:str) -> list[image_data]:
        arr = []
        try:
            with open(json_path, 'r') as the_file:
                for line in the_file:
                    obj_dict = json.loads(line)
                    obj = image_data.from_dict(**obj_dict)
                    obj.relative_path = obj_dict["relative_path"]
                    obj.objects = [object.from_dict(**o) for o in obj_dict["objects"]] 
                    for o in obj.objects:
                        o.box = box.from_dict(**o.box)

                    arr.append(obj)
        except Exception as e:
            print(e)
        
        return arr

    def __repr__(self):
        return json.dumps(self, default=obj_dict)


#------------------------------------------------------------------------------------------------------------------------------------------

def make_annotation_single(src_csv_path: str, src_line_parser_func: Function, dst_annotation_path: str, dst_write_mode:str) -> None:
    """Func make annotations from old csv format

    Args:
        src_csv_path (str): csv path
        src_line_parser_func (Function): Function that will be used to parser line string of csv file
                               These functions have to have bellow format:
                               + Argument: csv_path as string  --> e.g: C:\Data\gender.csv
                               + Return:   <absolute_image_path>, [(<object_box_x_y_w_h>), class_name1, class_name2,..., class_nameN] --> e.g:  "C:\Data\wman\1.jpg", [([0,0,200,300], wman, 30, back_head)]
        dst_annotation_path (str): Path to write annotation file
        dst_write_mode (str): Mode for writting annotation file
    """    
    if not os.path.exists(src_csv_path):
        print("File not found:", src_csv_path)
        return

    if src_line_parser_func is None:
        print("src_line_parser_func is None")
        return

    annotation_dir = to_linux_path(os.path.dirname(dst_annotation_path))
    all_data = []
    with open(src_csv_path, 'r') as the_file:
        for line in the_file:
            try:
                # parse line
                path, objs = src_line_parser_func(line)
                path = to_linux_path(path)
                isabs = os.path.isabs(path)

                # if is a absolute path then have to inside the annotation_dir
                if isabs:
                    if not path.startswith(annotation_dir):
                        print(f"Image is not include in the annotation dir: {path} --> skip")
                        continue          
                else:
                    path = os.path.join(annotation_dir, path)

                # check either path exist or not
                if not os.path.exists(path):
                    print(f"File not found {path} --> skip")
                    continue
       
                # get relative path
                rpath = to_linux_path(os.path.relpath(path, annotation_dir))
                objects = []

                for obj in objs:
                    obj_box= box(obj[0][0], obj[0][1], obj[0][2], obj[0][3])
                    class_names = [obj[i] for i in range(1, len(obj))]
                    objects.append(object(box= obj_box, class_names= class_names))
                
                data = image_data(relative_path=rpath, objects=objects)
                all_data.append(data)
            except Exception as e:
                print(e)
    image_dataset(all_data).serialize(dst_annotation_path, dst_write_mode)

def make_annotation_batch(src_csv_paths: list, src_line_parser_funcs: list[Function], dst_annotation_dir:str) -> None:
    """Func Make annotation for a list of csv file
     
    Usage is similar to the case of single file input function.
    Csv files maybe have diffent format itself.
    In this case, you can use the coorresponding parsing function for them.

    Args:
        src_csv_paths (list): list csv path
        src_line_parser_funcs (list[Function]): list func
        dst_annotation_dir (str): folder path
    """   
    if(not os.path.exists(dst_annotation_dir)):
        print("Directory not found: ", dst_annotation_dir)
        return
    
    if src_csv_paths is None or src_line_parser_funcs is None or len(src_csv_paths) != len(src_line_parser_funcs):
        print("csv_paths and csv_parsers must be not None and have the same size!")
        return

    annotation_name = "annotation"
    annotation_ext = ".txt"
    annotation_path = os.path.join(dst_annotation_dir, annotation_name + annotation_ext)

    if os.path.exists(annotation_path):
        strtime = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")
        os.rename(annotation_path, annotation_name + f"_bk_{strtime}" + annotation_ext)

    total = len(src_csv_paths)
    for i in range(total):
        print(f"[{i+1}/{total}]Processing {src_csv_paths[i]}")
        make_annotation_single(src_csv_paths[i], src_line_parser_funcs[i], annotation_path, "a+t")
    
    print(f"annotation file was saved to {annotation_path}")

def make_annotation_free(src_annotation_parser_func: Function, src_annotation_parser_input: object, dst_annotation_dir:str) -> None:
    """Write your own parsing function then pass into here

    Args:
        src_annotation_parser_func (Function): Function to parse your old annotation
                                                + Arguments: src_annotation_parser_input
                                                + Return:    image_dataset  instance object
        src_annotation_parser_input (object): Input argument to pass into the src_annotation_parser_func
        dst_annotation_dir (str): Directory contains dataset

    """    
    if(not os.path.exists(dst_annotation_dir)):
        print("Directory not found: ", dst_annotation_dir)
        return
    
    if src_annotation_parser_func is None:
        print("src_annotation_parser must be not None!")
        return

    annotation_name = "annotation"
    annotation_ext = ".txt"
    annotation_path = os.path.join(dst_annotation_dir, annotation_name + annotation_ext)

    if os.path.exists(annotation_path):
        strtime = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")
        os.rename(annotation_path,  os.path.join(dst_annotation_dir, annotation_name + f"_bk_{strtime}" + annotation_ext))
    
    dataset = src_annotation_parser_func(src_annotation_parser_input)
    if type(dataset) == image_dataset:
        dataset.serialize(annotation_path, "w")
        print(f"annotation file was saved to {annotation_path}")
    else:
        print(f"src_annotation_parser require return type {image_dataset} but got {type(dataset)}")

def to_yolo_anno(src_anno_path:str, src_class_index: int, dst_class_names: list[str],dst_class_ids:list[int]):
     dataset =  image_dataset.deserialize(src_anno_path)

     if dataset is not None and len(dataset) > 0:
        src_anno_dir = os.path.dirname(src_anno_path)
        for dt in dataset:
            try:
                # Parse data    
                txts = []
                for obj in dt.objects:
                    class_name = obj.class_names[src_class_index]
                    at = dst_class_names.index(class_name)
                    id = dst_class_ids[at]

                    txt = f"{id},{obj.box.x},{obj.box.y},{obj.box.w},{obj.box.h}\n"
                    txts.append(txt)
                
                # Write to text file
                image_path = os.path.join(src_anno_dir, dt.relative_path)
                image_name, image_ext = os.path.splitext(image_path)
                yolo_anno_path = image_name + ".txt"

                with open(yolo_anno_path, "w+t") as the_file:
                    the_file.writelines(txts)
            except Exception as e:
                 print(e)

def from_yolo_anno(src_data_dir:str):
    # scan all image files
    all_files = get_all_files(src_data_dir, [".jpg",".png",".bmp"])
    if all_files is None or len(all_files) < 1:
        print("No image file found!")
        return
    
    all_data = []
    total = len(all_files)
    for i in range(total):
        path = all_files[i]
        print(f"[{i}/{total}]: {path}")
        
        file_name, file_extension = os.path.splitext(path)
        ano_path = file_name + ".txt"
        if not os.path.exists(ano_path):
            print(f"File not found {ano_path}")
            continue
        
        img = cv2.imread(path)
        rpath = to_linux_path(os.path.relpath(path, src_data_dir))


        lines = None
        with open(ano_path) as file:
            lines = file.readlines()
            lines = [line.removesuffix("\r").removesuffix("\n").rstrip() for line in lines]
        if lines is None: 
            continue
        
        objects = []
        for line in lines:
            subs = line.split(" ")
            ok = []
            for s in subs:
                if len(s) > 0 and not str.isspace(s):
                    ok.append(s)
            subs = ok
            if len(subs) == 5:
                cx = float(subs[1]) * img.shape[1]
                cy = float(subs[2])* img.shape[0]
                w = float(subs[3])* img.shape[1]
                h = float(subs[4])* img.shape[0]
                objects.append(object(box(cx - w/2, cy-h/2, w, h), class_names=[subs[0]]))
        if len(objects) > 0:
            all_data.append(image_data(relative_path=rpath, objects=objects))
    
    return image_dataset(all_data)


def same_create_new_one():
    #1. Create data objects
    data1 = image_data(
        relative_path="shop1\1.jpg",
        objects=[
            object(box(0,0,10,10), class_names=["man","12","black_head"]),
            object(box(0,0,15,15), class_names=["man","42","font_head"]),
            object(box(0,0,20,20), class_names=["wman","90","front_head"]),
        ])

    data2 = image_data(
        relative_path="shop1\2.jpg",
        objects=[
            object(box(0,0,10,10), class_names=["man","12","black_head"]),
        ])
    
    #2. Serialize
    OK = image_dataset([data1, data2]).serialize("test.txt", "w")

    #3. Deserialize
    dataset =  image_dataset.deserialize("test.txt")
    print(dataset)
    print(dataset[0].relative_path)

def sample_convert_batch():
    # Convert from another one which have different format but using one text annotation file for all images
    '''
    Has 3 overloading functions you can use here:
    make_annotation_single() --> Convert a single csv file in old format
    make_annotation_batch()  --> Convert a batch of csv files in old format (maybe have different formats)
    make_annotation_free()   --> For a format does not using a single text file as annotation
    '''

    # 1. Put all your data into the same folder(annotation_dir) like bellow:
        # C:\DATASET
        # │   Dataset.txt
        # │
        # ├───Sub1
        # │   ├───Shop1
        # │   │       back (1).jpg
        # │   │       back (2).jpg
        # │   │       back (3).jpg
        # │   │
        # │   └───Shop2
        # │           back (1).jpg
        # │
        # ├───Sub2
        # │   └───Shop1
        # │           front (1).jpg
        # │           front (2).jpg
        # │
        # └───Sub3
        #     ├───Shop1
        #     │       front (1).jpg
        #     │
        #     ├───Shop2
        #     │       front (2).jpg
        #     │
        #     └───Shop3
    annotation_dir = r"D:\Data\Data_AgeGender\Age_Gender_Data_20211224"
    
    # 2. Collect all your old annotation files  
    csv_paths = [
    'D:\\Data\\Data_AgeGender\\Age_Gender_Data_20211224\\shop_thuoc_annotation_done\\#data.csv', 
    'D:\\Data\\Data_AgeGender\\Age_Gender_Data_20211224\\from_internet\\#data.csv',
    'D:\\Data\\Data_AgeGender\\Age_Gender_Data_20211224\\All-Age-Faces Dataset\\original images\\#data.csv', 
    'D:\\Data\\Data_AgeGender\\Age_Gender_Data_20211224\\Tara_shops_annotation_DONE_20211223\\#data.csv', 
    'D:\\Data\\Data_AgeGender\\Age_Gender_Data_20211224\\8_shop1_man1_result\\#data.csv'
    ]

    #3. Write your own function to parse each line for each csv file correspondingly
    def parser_line(strline:str) -> tuple[str, list[tuple[list[int], str, str]]]: 
        subs = strline.split(",")
        gender = subs[0]
        age = subs[1]
        apath = subs[2].replace("\r","").replace("\n","")

        return apath, [([0,0,0,0], gender, age)]

    csv_parsers = [parser_line, parser_line, parser_line, parser_line, parser_line]

    #4. Call function "make_annotation_batch" to gennerate the annotation file in standard format
    #   Annotation file will be saved into the "annotation_dir" and named as "annotation.txt" by default
    make_annotation_batch(csv_paths, csv_parsers, annotation_dir)

def sample_convert_single():
    annotation_path= r"C:\Users\Vmio-DungTv\Downloads\Age_Gender_Data_WithHat_20220104\anotation.txt"
    csv_path = r"C:\Users\Vmio-DungTv\Downloads\Age_Gender_Data_WithHat_20220104\Age_Gender_Data_WithHat_20220104.csv"

    def parser_line(strline:str) -> tuple[str, list[tuple[list[int], str, str]]]: 
        subs = strline.split(",")
        gender = subs[0]
        age = subs[1]
        apath = subs[2].replace("\r","").replace("\n","")

        return apath, [([0,0,0,0], gender, age)]

    make_annotation_single(src_csv_path=csv_path, src_line_parser_func=parser_line,dst_annotation_path=annotation_path, dst_write_mode= "w+t")

    

if __name__ == '__main__':

    # Sample usage
    # same_create_new_one()
    # sample_convert_batch()
    # sample_convert_single()

    # to_yolo_anno(
    # r"C:\Users\Vmio-DungTv\Downloads\Age_Gender_Data_20211224\anotation.txt",
    # 0,
    # ["0","1"],
    # [0,1]
    # )

    src_data_dir = r"C:\Users\Vmio-DungTv\Downloads\head-front-back.v1i.darknet"
    make_annotation_free(from_yolo_anno, src_data_dir, src_data_dir)

