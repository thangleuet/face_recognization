import shutil
import time
import cv2
import os
import tkinter as tk
from tkinter import ttk
import tkinter.font as font
from model.logger import miolog
from model.CameraCapture import CameraCapture
from PIL import Image, ImageTk
from controller.face_recog.facefaiss import FaceRecognition
import mediapipe as mp
import threading
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

DATASET = "D:\\PROJECT\\Face_recognize\\Mio-Ai-Cam-Viewer\\model\\models\\face_recog\\dataset"

class AddWindow():
    def __init__(self) -> None:
        # -----------UI element--------------
        # window
        self.add_window:tk.Toplevel = None
        self.aw_frame:tk.Frame = None
        # camera
        self.cam_frame:tk.Frame = None
        self.canvas_navi:tk.Canvas = None
        self.l_camera:tk.Label = None
        # thumbnail
        self.thumb_frame:tk.Frame = None
        self.canvas_thumb:tk.Canvas = None
        self.img_thumb:tk.Frame = None
        # control panel
        self.cp_frame:tk.Frame = None
        self.canvas_cp:tk.Canvas = None
        self.label_icon_cp:tk.Label = None
        self.label_cp:tk.Label = None
        self.sel:tk.StringVar = None
        self.cbox_cp:ttk.Combobox = None
        self.btn_submit:tk.Button = None
        self.btn_add:tk.Button = None
        self.btn_rm:tk.Button = None

        # other element
        self.cap = None
        self.image = None
        self.cap_fps:int = None
        self.video_captures:CameraCapture = None
        self.face_recog = None
        self.list_lb:list = None
        self.image_p:list = None
        self.is_change_face:bool = False

    def init_add_window(self,face_recog_model):
        self.face_recog = face_recog_model

        self.add_window = tk.Toplevel()
        self.add_window.title('Face Recognition')
        self.add_window.configure(background='white')
        self.add_window.minsize(1900, 1000)
        self.add_window.state('zoomed')

        self.add_window.protocol('WM_DELETE_WINDOW', self.close_add_window)

        self.add_window.update()

        add_width_screen = self.add_window.winfo_width()
        add_height_screen = self.add_window.winfo_height()
        # print(add_width_screen)
        # print(add_height_screen)

        self.aw_frame = tk.Frame(self.add_window)
        self.aw_frame.grid(row=0, column=0, stick='news')

        self.add_window.rowconfigure(0, weight=1)
        self.add_window.columnconfigure(0, weight=1)

        self.aw_frame.rowconfigure(0, weight=2)
        self.aw_frame.columnconfigure(0, weight=2)
        self.aw_frame.rowconfigure(1, weight=1)
        self.aw_frame.columnconfigure(1, weight=1)
        self.aw_frame.update()

        # Create a camera view
        self.cam_frame = tk.Frame(self.aw_frame,
                            width=int(add_width_screen*2/3),
                            height=int(add_height_screen*2/3))
        self.cam_frame.grid(row=0,column=0,sticky="news")
        self.cam_frame.rowconfigure(0, weight=1)
        self.cam_frame.columnconfigure(0, weight=1)
        self.cam_frame.update()
        self.l_camera = tk.Label(self.cam_frame, bg="black", highlightbackground="#45fc03", highlightthickness=3)
        self.l_camera.grid(row=0,column=0, sticky="news", padx=10, pady=10)
        self.show_cam()



        # create thumbnail view
        self.thumb_frame = tk.Frame(self.aw_frame,
                            width=int(add_width_screen),
                            height=int(add_height_screen*1/3))
        self.thumb_frame.grid(row=1,columnspan=2,sticky="news")
        self.thumb_frame.rowconfigure(0,weight=1)
        self.thumb_frame.columnconfigure(0,weight=1)
        self.thumb_frame.update()
        self.canvas_thumb = tk.Canvas(self.thumb_frame, bg="white", highlightbackground="gray", highlightthickness=2)
        self.canvas_thumb.grid(row=0,column=0,padx=5,pady=5,sticky="news")
        self.img_thumb = tk.Frame(self.thumb_frame,bg="white")
        self.img_thumb.grid(row=0,column=0,padx=10,pady=10,sticky="news")

        # creare control panel view
        self.cp_frame = tk.Frame(self.aw_frame,
                            width=int(add_width_screen*1/3),
                            height=int(add_height_screen*2/3))
        self.cp_frame.grid(row=0,column=1,sticky="news")
        self.cp_frame.rowconfigure(0, weight=1)
        self.cp_frame.rowconfigure(1, weight=1)
        self.cp_frame.columnconfigure(0, weight=1)
        self.cp_frame.update()
        self.canvas_cp = tk.Canvas(self.cp_frame, bg='white', highlightbackground="blue", highlightthickness=2)
        self.canvas_cp.grid(rowspan=2,column=0, padx=5, pady=5, sticky="news")
        # vmio_image = Image.open("D:/PROJECT/Face_recognize/Mio-Ai-Cam-Viewer/icon/logo_bold_crop.png")
        # size = int(self.cp_frame.winfo_height()/3)
        # vmio_image = vmio_image.resize((size,size),Image.ANTIALIAS)
        #vmio_icon = ImageTk.PhotoImage(vmio_image)
        #self.label_icon_cp = tk.Label(self.cp_frame, image=vmio_icon, bg="white")
        #self.label_icon_cp.grid(row=0,column=0,padx=5,pady=20, sticky="n")
        self.get_list_labels()

        self.label_cp = tk.Label(self.cp_frame,bg="white", text="Choose existing label or type a new one",
                                    foreground="blue",font=("Roboto",15,"bold"))
        self.label_cp.grid(row=1,column=0,sticky="n",pady=10)
        self.sel = tk.StringVar()
        self.cbox_cp = ttk.Combobox(self.cp_frame, values=self.list_lb,
                                    font=('Roboto',16,'bold'),
                                    width=25,textvariable=self.sel)
        self.cbox_cp.grid(row=1,column=0,sticky="n",pady=50)
        self.btn_submit = tk.Button(self.cp_frame,bg='blue',highlightbackground='black',highlightthickness=2,
                                        text="Get",font=("Roboto",18,"bold"),foreground="white",
                                        width=15, command=lambda:self.get_label_images())
        self.btn_submit.grid(row=1,column=0,sticky='n',pady=100)
        self.btn_add = tk.Button(self.cp_frame,bg='green',highlightbackground='black',highlightthickness=2,
                                        text="Add",font=("Roboto",18,"bold"),foreground="white",
                                        width=15, command=lambda:self.add_label_images())
        self.btn_add.grid(row=1,column=0,sticky='n',pady=160)
        self.btn_rm = tk.Button(self.cp_frame,bg='red',highlightbackground='black',highlightthickness=2,
                                        text="Delete",font=("Roboto",18,"bold"),foreground="white",
                                        width=15,command=lambda:self.delete_label_images())
        self.btn_rm.grid(row=1,column=0,sticky='n',pady=220)



        self.add_window.mainloop()

    def delete_label_images(self):
        try:
            cur_label = self.sel.get()
            if not cur_label:
                print("empty label")
                return
            if self.face_recog.delete(cur_label):
                if os.path.isdir(f"{DATASET}/{cur_label}"):
                    shutil.rmtree(f"{DATASET}/{cur_label}")
                self.list_lb.remove(cur_label)
                self.cbox_cp.configure(values=self.list_lb)
                self.sel.set("")
                self.get_label_images()
                self.is_change_face = True
                miolog.info(f'[Add New Face]: delete {cur_label} from dataset')
        except Exception as e:
            print(e)

    def add_label_images(self):
        try:
            cur_label = self.sel.get()
            img_name = int(time.time())
            img = self.image
            if not cur_label:
                print("empty label")
                return
            list_box =self.face_recog.get_face_box(img)
            if len(list_box)>0:
                x,y,w,h = self.face_recog.get_bigest_box(list_box)
                img = img[max(0,y-100):min(y+h+100,int(self.aw_frame.winfo_height()*2/3-50)),max(0,x-50):min(x+w+50,int(self.aw_frame.winfo_width()*2/3-100))]
                if self.face_recog.train(img,cur_label):
                    if not os.path.isdir(f"{DATASET}/{cur_label}"):
                        os.mkdir(f"{DATASET}/{cur_label}")
                        self.list_lb.append(cur_label)
                        self.cbox_cp.configure(values=self.list_lb)
                    cv2.imwrite(f"{DATASET}/{cur_label}/{img_name}.jpg",img)
                    self.get_label_images()
                    self.is_change_face = True
                    miolog.info(f'[Add New Face]: trained {cur_label}')
        except Exception as e:
            print(e)


    def get_label_images(self):
        try:
            self.remove_grid_slave(self.img_thumb)
            cur_label = self.sel.get()
            if not cur_label:
                print("empty label")
                return
            self.image_p = glob.glob(f"{DATASET}/{cur_label}/*.jpg")
            img_list = list()

            for i in range(len(self.image_p)):
                print(self.image_p)
                image = Image.open(self.image_p[i])
                w,h = image.size
                ratio = self.aw_frame.winfo_height()/(3*h)
                new_w = int(w*ratio*2/3)
                new_h = int(h*ratio*2/3)
                # print(new_h)
                image = image.resize((new_w,new_h),Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                img_list.append(tk.Label(self.img_thumb))
                img_list[i].imgtk = photo
                img_list[i].configure(image=photo)
                img_list[i].grid(row=0,column=i,sticky='w',padx=10,pady=5)
        except Exception as e:
            print(e)

    def remove_grid_slave(self,grid):
        for wg in grid.grid_slaves():
            wg.destroy()

    def get_list_labels(self):
        self.list_lb = []
        for folder in glob.glob(f"{DATASET}/*", recursive=True):
            self.list_lb.append(folder.split('\\')[-1])
        print(self.list_lb)

    def show_cam(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        self.cap_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.video_captures = CameraCapture(self.cap)
        self.show_frame()

    def show_frame(self):
        try:
            ret, frame = self.video_captures.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.image = frame
                height, width = frame.shape[:2]
                scale_ratio = (self.l_camera.winfo_height()) / height
                frame = cv2.resize(frame, (0,0), fx = scale_ratio, fy = scale_ratio)
                img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img1 = self.draw_face_mesh(img_rgb=img1)
                img1 = ImageTk.PhotoImage(Image.fromarray(img1))
                self.l_camera.imgtk = img1
                self.l_camera.configure(image=img1)
                self.l_camera.grid(row=0,column=0, padx=5, pady=5)
            self.add_window.after(int(1000/self.cap_fps), self.show_frame)
        except Exception as e:
            print(e)

    def stop_vid(self):
        try:
            if self.cap:
                self.video_captures.close(1)
                self.cap.release()
                cv2.destroyAllWindows()
                if self.l_camera is not None:
                    self.l_camera.after(1000, self.l_camera.grid_forget())
        except Exception as e:
            print(e)

    def close_add_window(self):
        self.stop_vid()
        miolog.info("[Add New Face] close window")
        self.add_window.destroy()

    def draw_face_mesh(self, img_rgb):
        try:
            with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
                img_rgb.flags.writeable = False
                results = face_mesh.process(img_rgb)
                img_rgb.flags.writeable = True
                image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
                img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return img1
        except Exception as e:
            print(e)