import cv2
import tkinter as tk
import tkinter.font as font
import threading

from PIL import Image, ImageTk
from model.logger import miolog
from model.CameraCapture import CameraCapture
from controller.face_recog.facefaiss import FaceRecognition, draw_target, welcomespeech
from view.add_face_ui import AddWindow

class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom


main_root = None
# ***** VARIABLES *****
# use a boolean variable to help control state of time (running or not running)
running = False
# time variables initially set to 0
hours, minutes, seconds = 0, 0, 0

is_facerecog_model = False
is_start_camera = True
is_record_camera = True
cam_on = False
cap = None
cap_fps = None
video_captures = None
l_camera = None
l_record = None
l_stopwatch = None

list_lb = []
newwindow = None

# ***** NOTES ON GLOBAL *****
# global will be used to modify variables outside functions
# another option would be to use a class and subclass Frame

# ***** FUNCTIONS *****
# start, pause, and reset functions will be called when the buttons are clicked
# start function
def start_record():
    global running
    if not running:
        update()
        running = True


# reset function
def reset_record():
    global running, l_stopwatch
    if running:
        # cancel updating of time using after_cancel()
        l_stopwatch.after_cancel(update_time)
        running = False
    # set variables back to zero
    global hours, minutes, seconds
    hours, minutes, seconds = 0, 0, 0
    # set label back to zero
    l_stopwatch.config(text='00:00:00')


# update stopwatch function
def update():
    # update seconds with (addition) compound assignment operator
    global hours, minutes, seconds, l_stopwatch
    seconds += 1
    if seconds == 60:
        minutes += 1
        seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    # format time to include leading zeros
    hours_string = f'{hours}' if hours > 9 else f'0{hours}'
    minutes_string = f'{minutes}' if minutes > 9 else f'0{minutes}'
    seconds_string = f'{seconds}' if seconds > 9 else f'0{seconds}'
    # update timer label after 1000 ms (1 second)
    l_stopwatch.config(text=hours_string + ':' + minutes_string + ':' + seconds_string)
    # after each second (1000 milliseconds), call update function
    # use update_time variable to cancel or pause the time using after_cancel
    global update_time
    update_time = l_stopwatch.after(1000, update)


def process_face(frame):
    global list_lb
    try:
        list_box = face_recog.get_face_box(frame)
        # print(list_box)
        # print(len(list_box))
        if len(list_box) > 0:
            for (x,y,w,h) in list_box:
                face = frame[y:y+h, x:x+w]
                # if i%60==0:
                label, score = face_recog.face_recog(face)
                if label == 'unknown':
                    color_lb = (255,0,0)
                elif label == 'None':
                    color_lb = (87, 86, 86)
                else:
                    color_lb = (0,0,255)
                frame = draw_target(frame, (x,y,w,h))
                w,h = cv2.getTextSize(f"{label} [{int(score)}%]",cv2.FONT_HERSHEY_DUPLEX,0.7,2)[0]
                cv2.rectangle(frame,(x,y-5),(x+w,y-h-10),color_lb,-1)
                frame = cv2.putText(frame, f"{label} [{int(score)}%]", (x,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LSD_REFINE_STD)
                if label not in list_lb:
                    list_lb.append(label)
                    thread = threading.Thread(target=welcomespeech, args=[label,])
                    thread.start()
        else:
            list_lb = []
    except Exception as e:
        print(e)

def show_frame():
    global l_camera
    if cam_on:
        list_count_people = []
        ret, frame = video_captures.read()
        if ret:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            scale_ratio = (main_root.winfo_height() - 80) / height
            frame = cv2.resize(frame, (0,0), fx = scale_ratio, fy = scale_ratio)
            img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
         
            _thread_facerecog = threading.Thread(target=process_face, args=(img1,))
            if is_facerecog_model:
                _thread_facerecog.start()
          
            if _thread_facerecog.is_alive():
                _thread_facerecog.join()


            img1 = ImageTk.PhotoImage(Image.fromarray(img1))
            l_camera.imgtk = img1
            l_camera.configure(image=img1)
            l_camera.grid(row=1, columnspan=2, sticky='ns', padx=10, pady=10)

        main_root.after(int(1000/cap_fps), show_frame)


def stop_vid():
    global cam_on, l_camera
    cam_on = False

    if cap:
        video_captures.close(1)
        cap.release()
        cv2.destroyAllWindows()
        if l_camera is not None:
            l_camera.after(1000, l_camera.grid_forget())


def is_camera(btn):
    miolog.info("Click open camera")
    global is_start_camera, cam_on, cap, cap_fps, video_captures, l_camera, canvas_camera, face_recog, newwindow

    if newwindow.is_change_face:
        print("change face recog")
        # face_recog = newwindow.face_recog
        newwindow.is_change_face = False

    if is_start_camera:
        miolog.info("Show frame")

        is_start_camera = False

        # Creating a photoimage object to use image
        photo = tk.PhotoImage(file="./icon/icons8-square-48.png")
        btn.config(image=photo, text="STOP")
        btn.image = photo
        canvas_camera.configure(highlightbackground="#45fc03")

        # init show frame
        stop_vid()
        l_camera.grid()
        cam_on = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_captures = CameraCapture(cap)
        show_frame()

    else:
        miolog.info("Click stop camera")
        is_start_camera = True

        # Creating a photoimage object to use image
        photo = tk.PhotoImage(file="./icon/icons8-play-64.png")
        btn.config(image=photo, text="START")
        btn.image = photo
        canvas_camera.configure(highlightbackground="red")

        miolog.info("Stop camera")
        # init off frame
        stop_vid()


def is_record(btn, width_screen):
    global is_record_camera, l_record, l_stopwatch

    miolog.info("Click record video")
    if is_record_camera:
        miolog.info("Start record video")
        is_record_camera = False

        photo = tk.PhotoImage(file="./icon/icon_stop_record.png")
        btn.config(image=photo)
        btn.image = photo

        l_record.grid(row=0, column=1, padx=290, pady=10, sticky='en')
        l_stopwatch.grid(row=0, column=1, padx=290, pady=10, sticky='es')

        start_record()
    else:
        miolog.info("Stop record video")
        is_record_camera = True

        photo = tk.PhotoImage(file="./icon/icon_start_record.png")
        btn.config(image=photo)
        btn.image = photo

        l_record.grid_forget()
        l_stopwatch.grid_forget()

        reset_record()


def open_add_window(btn):
    global is_start_camera, face_recog, newwindow
    is_start_camera = True
    # Creating a photoimage object to use image
    photo = tk.PhotoImage(file="./icon/icons8-play-64.png")
    btn.config(image=photo, text="START")
    btn.image = photo
    canvas_camera.configure(highlightbackground="red")
    stop_vid()
    # newwindow = AddWindow()
    miolog.info("[Add New Face] open window")
    newwindow.init_add_window(face_recog)


def is_face_recog(btn):
    global is_facerecog_model, newwindow, face_recog, list_lb
    if is_facerecog_model == True:
        is_facerecog_model = False
        list_lb.clear()
        btn.configure(bg = 'SystemButtonFace')
        miolog.info('disable face recog')
    else:
        is_facerecog_model = True
        face_recog.load_DB()
        btn.configure(bg = "#8af77e")
        miolog.info('enable face recog')

def full_screen_mode(event):
    main_root.attributes('-fullscreen', True)


def out_full_screen(event):
    main_root.attributes('-fullscreen', False)

def menu_popup(popup, button):
    # display the popup menu
    try:
        x = button.winfo_rootx()
        y = button.winfo_rooty()
        popup.tk_popup(x,y+60, 0)
    finally:
        #Release the grab
        popup.grab_release()




def init_component(list_model):
    global main_root, l_camera, l_record, l_stopwatch, face_recog, list_lb
    global popup, newwindow, canvas_camera


    miolog.info('---------Init Main UI---------')
    main_root = tk.Tk()
    main_root.title('Face Recognition')
    main_root.configure(background='white')
    main_root.minsize(850, 700)
    main_root.state('zoomed')


    # ico = Image.open('./icon/logo_bold_crop.png')
    # photo = ImageTk.PhotoImage(ico)
    # main_root.iconphoto(False, photo)

    main_root.state('zoomed')
    main_root.bind('<F11>', full_screen_mode)
    main_root.bind('<Escape>', out_full_screen)
    # FullScreenApp(main_root)
    # main_root.attributes('-fullscreen', True)
    main_root.update()

    try:
        face_recog = list_model[0]
    except Exception as e:
        face_recog = None
        miolog.exception(e, "[init_ui] init_component")

    newwindow = AddWindow()

    width_screen = main_root.winfo_width()
    height_screen = main_root.winfo_height()

    miolog.info('width_screen: {}'.format(width_screen))
    miolog.info('height_screen: {}'.format(height_screen))

    frame = tk.Frame(main_root)
    frame.grid(row=0, column=0, stick='news')

    main_root.rowconfigure(0, weight=1)
    main_root.columnconfigure(0, weight=1)

    # Create a canvas widget
    canvas_navi = tk.Canvas(frame, bg='white', highlightbackground="gray", highlightthickness=1, width=width_screen - 8,
                        height=65)

    canvas_navi.grid(row=0, padx=5, pady=5,columnspan=2)

    # -------------------Button Start-----------------------
    photo = tk.PhotoImage(file="./icon/icons8-play-64.png")

    btn_open = tk.Button(frame, text='START', image=photo, compound="left", borderwidth=0, height=54,
                         highlightbackground="gray", highlightthickness=2, command=lambda: is_camera(btn_open))
    # btn_open.image = photo
    btn_open['font'] = font.Font(weight="bold")
    btn_open.grid(row=0, column=0, padx=10, sticky='w')

    # -------------------Button Setting-----------------------
    popup = tk.Menu(main_root, tearoff=0)
    popup.add_command(label='Add Face', command=lambda:open_add_window(btn_open))
    popup.add_separator()
    popup.add_command(label='Setting')
    photo_setting = tk.PhotoImage(file="./icon/icons8-setting-64.png")
    btn_setting = tk.Button(frame, image=photo_setting, borderwidth=0, height=60,
                            highlightbackground="white", command=lambda: menu_popup(popup, btn_setting))

    # btn_setting.image = photo_setting
    # btn_setting.place(x=width_screen - 75, y=2)
    btn_setting.grid(row=0, column=1, padx=10, sticky='e')
    # -------------------Button Distance-----------------------
    photo_face_recog = tk.PhotoImage(file="./icon/icons8-face-id-64.png")
    btn_face_recog = tk.Button(frame, image=photo_face_recog, borderwidth=0, height=60,
                             highlightbackground="white", command=lambda: is_face_recog(btn_face_recog))
    btn_face_recog.grid(row=0, column=1, padx=80, sticky='e')
    # -------------------Button Distance-----------------------
    # photo_distance = tk.PhotoImage(file="./icon/distanceMeasurement64x64-02.png")
    # btn_distance = tk.Button(frame, image=photo_distance, borderwidth=0, height=60,
    #                          highlightbackground="white", command=lambda: is_positioning(btn_distance))
    # # btn_distance.image = photo_distance
    # # btn_distance.place(x=width_screen - 150, y=2)
    # btn_distance.grid(row=0, column=1, padx=150, sticky='e')

    # # -------------------Button Face-----------------------
    # photo_face = tk.PhotoImage(file="./icon/drag-gender-neutral.png")
    # btn_face = tk.Button(frame, image=photo_face, borderwidth=0, height=60,
    #                      highlightbackground="white", command=lambda: is_age_gender(btn_face))
    # btn_face.image = photo_face
    # btn_face.place(x=width_screen - 225, y=2)
    #btn_face.grid(row=0, column=1, padx=220, sticky='e')

    # -------------------Button Record-----------------------
    photo_record = tk.PhotoImage(file="./icon/icon_start_record.png")
    btn_record = tk.Button(frame, image=photo_record, borderwidth=0, height=60,
                           highlightbackground="white", command=lambda: is_record(btn_record, width_screen))
    # btn_record.image = photo_record
    # btn_record.place(x=width_screen - 325, y=2)
    btn_record.grid(row=0, column=1, padx=150, sticky='e')

    l_record = tk.Label(frame, text="Recording...", highlightbackground="white", bg="white", fg="#e0563a")
    # l_record.grid(row=0, column=1, padx=290, pady=10, sticky='en')

    # label to display time
    l_stopwatch = tk.Label(frame, text='00:00:00', font=('Arial', 20), highlightbackground="white", bg="white", fg="#f76248")
    # l_stopwatch.grid(row=0, column=1, padx=290, pady=10, sticky='es')

    # -------------------Init Camera----------------------
    canvas_camera = tk.Canvas(frame, bg='black', highlightbackground="red", highlightthickness=2, width=width_screen - 8,
                              height=height_screen - 75)
    # canvas_camera.pack()
    canvas_camera.grid(row=1, columnspan=2, padx=5, pady=5, sticky='news')

    border_camera = tk.LabelFrame(frame, bg="black")
    # border_camera.place(relx=0.5, rely=0.5, anchor="center")
    border_camera.grid(row=1, columnspan=2, padx=5, pady=5)

    l_camera = tk.Label(frame, bg="black", width=width_screen - 200)
    l_camera.grid(row=1, columnspan=2, sticky='ns', padx=10, pady=10)

    frame.rowconfigure(0, weight=0)
    frame.rowconfigure(1, weight=1)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)

    main_root.mainloop()


