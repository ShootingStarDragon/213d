#things for subprocess to work has to be outside the if __name__ == '__main__' guard so subprocesses have access to it
import cv2
import numpy as np
import time
import sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def cv_haar_cascade_async(*args):
    '''
    reference: https://stackoverflow.com/questions/70805922/why-does-the-haarcascades-does-not-work-on-opencv
    '''
    try:
        ret_var = args[0]
        frame_var =  args[1]
        shared_dict_var = args[2]
        frame_int_var = args[3]

        if ret_var:
            #they resized it to be less laggy:
            w_size = (700,500)
            frame_var = cv2.resize(frame_var,w_size)

            gray = cv2.cvtColor(frame_var,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame_var, (x, y), (x+w, y+h), (0, 255, 0), 2)
            buf1 = cv2.flip(frame_var, 0)
            buf1 = cv2.flip(buf1, 1)
            shared_dict_var[frame_int_var] = buf1
            # print("timedelta!", time_og, time_end - time_og, 1/60, frame_int_var, flush= True)
            # https://stackoverflow.com/questions/58614788/how-do-i-get-the-multiprocessing-running/58615142#58615142
        sys.stdout.flush() #you need this line to get python to have no buffer else things get laggy, like for the haarcascades example
    except Exception as e:
        print("exception as e cv_async", e, flush=True ) #same as sys.stdout.flush()

def cv_sepia_async(*args):
    '''
    reference: https://medium.com/dataseries/designing-image-filters-using-opencv-like-abode-photoshop-express-part-2-4479f99fb35
    '''
    try:
        ret_var = args[0]
        frame_var =  args[1]
        shared_dict_var = args[2]
        frame_int = args[3]
        if ret_var:
            img = np.array(frame_var, dtype=np.float64) # converting to float to prevent loss
            img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
            img[np.where(img > 255)] = 255 # normalizing values greater than 255 to 255
            img = np.array(img, dtype=np.uint8) # converting back to int
            buf1 = cv2.flip(img, 0)
            buf1 = cv2.flip(buf1, 1)
            shared_dict_var[frame_int] = buf1
    except Exception as e:
        print("exception as e cv_async", e, flush=True )

backSub = cv2.createBackgroundSubtractorMOG2()

def cv_backSub_async(*args):
    '''
    reference: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    '''
    try:
        ret_var = args[0]
        frame_var =  args[1]
        shared_dict_var = args[2]
        frame_int_var = args[3]
        if ret_var:
            fgMask = backSub.apply(frame_var)
            fgMask = cv2.cvtColor(fgMask,cv2.COLOR_GRAY2RGB)
            buf1 = cv2.flip(fgMask, 0)
            buf1 = cv2.flip(buf1, 1)
            shared_dict_var[frame_int_var] = buf1
        sys.stdout.flush() #you need this line to get python to have no buffer else things get laggy, like for the haarcascades example
    except Exception as e:
        print("exception as e cv_async", e, flush=True ) #same as sys.stdout.flush()

def cv_canny_async(*args):
    '''
    reference: https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
    '''
    try:
        ret_var = args[0]
        frame_var =  args[1]
        shared_dict_var = args[2]
        frame_int_var = args[3]
        if ret_var:
            ratio = 3
            kernel_size = 3
            low_threshold = 50
            img_blur = cv2.blur(cv2.cvtColor(frame_var,cv2.COLOR_RGB2GRAY), (3,3))
            detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
            mask = detected_edges != 0
            dst = frame_var * (mask[:,:,None].astype(frame_var.dtype))
            # dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2RGB)
            buf1 = cv2.flip(dst, 0)
            buf1 = cv2.flip(buf1, 1)
            shared_dict_var[frame_int_var] = buf1

# ratio = 3
# kernel_size = 3
# def CannyThreshold(val):
#     low_threshold = val
#     img_blur = cv.blur(src_gray, (3,3))
#     detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
#     mask = detected_edges != 0
#     dst = src * (mask[:,:,None].astype(src.dtype))
#     cv.imshow(window_name, dst)


        sys.stdout.flush() #you need this line to get python to have no buffer else things get laggy, like for the haarcascades example
    except Exception as e:
        print("exception as e cv_async", e, flush=True ) #same as sys.stdout.flush()

def cv_async(*args):
    try:
        ret_var = args[0]
        frame_var =  args[1]
        shared_dict_var = args[2]
        frame_int = args[3]
        # print("this work?", ret_var, type(frame_var), flush = True)
        if ret_var:
            # print("this work?2", ret_var, type(frame_var), flush = True)
            buf1 = cv2.flip(frame_var, 0)
            buf1 = cv2.flip(buf1, 1)
            shared_dict_var[frame_int] = buf1
    except Exception as e:
        print("exception as e cv_async", e, flush=True )

def cv_asyncded(*args):
    try:
        ret, frame = stream.read(0)
        print("so is this running?", ret, flush=True)
        if ret:
            #here just put the frame to the shared dictionary
            shared_dict[frame_int] = frame.tobytes()
            # buf1 = cv2.flip(frame, 0)
            # buf1 = cv2.flip(buf1, 1)
            # buf = buf1.tobytes()
            # texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            # texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # App.get_running_app().root.get_screen('start_screen_name').ids["image_textureID"].texture = texture1
        else:
            print(f"no cv2 capture at index {device_index}", flush=True) 
    except Exception as e:
        print("exception as e cv_async", e, flush=True )
        
if __name__ == '__main__':
    import kivy
    kivy.require('2.1.0') # replace with your current kivy version !

    from kivy.app import App
    from kivy.lang import Builder
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.graphics.texture import Texture
    from kivy.clock import Clock

    from typing import Optional
    from models.models import texture_format
    from datetime import datetime

    import multiprocessing as FCVA_mp
    FCVA_mp.freeze_support()
    #need pool to be in global namespace sadly https://stackoverflow.com/a/32323448
    #  FCVApool = FCVA_mp.Pool(FCVA_mp.cpu_count())
    FCVApool = FCVA_mp.Pool(4)
    shared_mem_manager = FCVA_mp.Manager()
    shared_analysis_dict = shared_mem_manager.dict()
    
    #just in case somebody is using textures before making the app:
    '''
    If you need to create textures before the application has started, import
        Window first: `from kivy.core.window import Window`
    '''
    from kivy.core.window import Window

    class FastCVApp(App):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.device_index = 0
            self.frame_int = 0
            #remember that the KV string IS THE ACTUAL FILE AND MUST BE INDENTED PROPERLY TO THE LEFT!
            self.KV_string = '''
<FCVA_screen_manager>:
    id: FCVA_screen_managerID
    StartScreen:
        id: start_screen_id
        name: 'start_screen_name'
        manager: 'FCVA_screen_managerID'

<StartScreen>:
    id: start_screen_id
    BoxLayout:
        orientation: 'vertical'
        Image:
            id: image_textureID
        Label:
            text: "hello world!"

FCVA_screen_manager: #remember to return a root widget
                '''

        def build(self):
            #only build in the main process:
            if __name__ == '__main__':
                self.title = "Fast CV App v0.1.0 by Pengindoramu"
                build_app_from_kv = Builder.load_string(self.KV_string)
                Window.bind(on_request_close=self.on_request_close)
                return build_app_from_kv
            else:
                print("Are you sure you're running from __main__? Spawning a subprocess from a subprocess is not ok.")
        
        def on_start(self):
            # opening a camera here is laggy and delays the startup time so start after the gui is loaded with this
            print("schedule interval 0", datetime.now().strftime("%H:%M:%S"))
            Clock.schedule_once(self.init_cv, 0)

        def init_cv(self, *args):
            self.stream = cv2.VideoCapture(self.device_index)
            print("what is stream type?", type(self.stream))
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            print("ret, frame!", datetime.now().strftime("%H:%M:%S"))
            print("fps of stream?", self.fps)
            # Clock.schedule_interval(self.cv_func, 1/60)
            Clock.schedule_interval(self.blit_from_shared_memory, 1/(self.fps))
            '''
            question:
                WHEN DO YOU ANALYZE FRAMES?
                I already have a schedule interval that blits to texture
                obv ans:
                    in the blit schedule interval you already call for the multiprocessing of the stream
                    cons:
                        this will make it a bit slower (noticably slower?)
                        also how will you ever sync with media? <-- this is a problem I have currently anyways...
            '''

        def blit_from_shared_memory(self, *args):
            ret, frame = self.stream.read(0)
            self.what = FCVApool.apply_async(cv_canny_async, args=(ret, frame, shared_analysis_dict, self.frame_int)) 
            #THIS WORKS: self.what = FCVApool.apply_async(cv_backSub_async, args=(ret, frame, shared_analysis_dict, self.frame_int)) 
            #THIS WORKS: self.what = FCVApool.apply_async(cv_haar_cascade_async, args=(ret, frame, shared_analysis_dict, self.frame_int)) 
            #THIS WORKS: self.what = FCVApool.apply_async(cv_sepia_async, args=(ret, frame, shared_analysis_dict, self.frame_int)) 
            #THIS WORKS: self.what = FCVApool.apply_async(cv_async, args=(ret, frame, shared_analysis_dict, self.frame_int)) 
            #problem is I don't think you can pickle the stream for multiprocessing (it's a tuple, idk if you can send tuples in a tuple), so send the frame instead
            # https://stackoverflow.com/questions/17872056/how-to-check-if-an-object-is-pickleable
            # import dill
            # print("dill pickles!", dill.pickles(self.stream)) #says false, so I can't send the stream, but I can still send the individual frame
            self.frame_int += 1
            # print("is this at least mp?", self.what)
            # print("#check if there's something in shared memory:", len(shared_analysis_dict))
            if len(shared_analysis_dict) > 0:
                #get the max key
                max_key = max(shared_analysis_dict.keys())
                # print("maxkey?", max_key)
                frame = shared_analysis_dict[max_key]
                buf = frame.tobytes()
                #blit to texture
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                App.get_running_app().root.get_screen('start_screen_name').ids["image_textureID"].texture = texture1
                #after blitting delete if dict has more than 10 frames:
                if len(shared_analysis_dict) > 5:
                    min_key = min(shared_analysis_dict.keys())
                    del shared_analysis_dict[min_key]
        
        def on_request_close(self, *args):
            self.stream.release()
            pass

        def cv_func(self, *args):
            ret, frame = self.stream.read(0)
            if ret:
                buf1 = cv2.flip(frame, 0)
                buf1 = cv2.flip(buf1, 1)
                buf = buf1.tobytes()
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                App.get_running_app().root.get_screen('start_screen_name').ids["image_textureID"].texture = texture1
            else:
                print("no cv2 capture at index 0")
            

    class FCVA_screen_manager(ScreenManager):
        pass

    class StartScreen(Screen):
        pass

    # https://realpython.com/primer-on-python-decorators/
    # # https://stackoverflow.com/questions/5929107/decorators-with-parameters
    # def CV_function(texture_ref = "None", format: Optional[texture_format] = "RGB"):
    #     # if texture_ref == "None":
    #     #     texture_ref = App.get_running_app().root.get_screen('start_screen_name').ids["start_screen_id"].ids["image_textureID"]
    #     print("texture_ref worksA?", texture_ref, type(texture_ref))
    #     def wrapper(func, *args, **kwargs):
    #         def wrapped_f(*args, **kwargs):

    #             # final_texture = func(*args, **kwargs)
    #             # #if format is not specified, default to RGB
    #             try:
    #                 print("texture_ref worksB?", texture_ref, type(texture_ref))
    #                 if texture_ref == "None":
    #                     #if I use this app MUST be App so that everything works...
    #                     #next line just kills everything... "local variable 'texture_ref' referenced before assignment" when trying to exit from kivy window unsure why
    #                     #I KNOW WHY BECAUSE I RUN IT AFTER app.run() but after I close it doesn't exist anymore...
    #                     texture_ref = App.get_running_app().root.get_screen('start_screen_name').ids["start_screen_id"].ids["image_textureID"]
    #                     print("x")
    #             except Exception as e:
    #                 print(e)

    #             # buf = buf1.tobytes()
    #             # texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt=texture_format.format) 
    #             # texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    #             # self.ids["textureB"].texture = texture1

    #         return wrapped_f
    #     return wrapper

    # @CV_function("None")
    # def testfunc():
    #     pass

    #usage, just decorate their CV_function like so: @CV_function
    FastCVApp().run()
