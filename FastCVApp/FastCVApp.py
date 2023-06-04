# so that main and subprocesses have access to this since it's not under if __name__ is main
import cv2
import time
import os, sys
import numpy as np
from FCVAutils import fprint
import blosc2

def open_kivy(*args):
    # infinite recursion bug when packaging with pyinstaller with no console: https://github.com/kivy/kivy/issues/8074#issuecomment-1364595283
    if sys.__stdout__ is None or sys.__stderr__ is None:
        os.environ["KIVY_NO_CONSOLELOG"] = "1"
    from kivy.app import App
    from kivy.lang import Builder
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.graphics.texture import Texture
    from kivy.clock import Clock
    from kivy.modules import inspector
    from kivy.core.window import Window
    from kivy.uix.button import Button

    class MainApp(App):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            shared_metadata_dict = self.shared_metadata_dictVAR
            kvstring_check = [
                shared_metadata_dict[x]
                for x in shared_metadata_dict.keys()
                if x == "kvstring"
            ]
            if len(kvstring_check) != 0:
                self.KV_string = kvstring_check[0]
            else:
                # remember that the KV string IS THE ACTUAL FILE AND MUST BE INDENTED PROPERLY TO THE LEFT!
                self.KV_string = f"""
#:import kivy.app kivy.app
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
        id: mainBoxLayoutID
        Image:
            id: image_textureID
        Slider:
            id: vidsliderID
            min: 0
            max: {self.framelength} #should be 30*total_seconds
            step: 1
            value_track: True
            value_track_color: 1, 0, 0, 1
            size_hint: (1, 0.1)
            orientation: 'horizontal'
        BoxLayout:
            id: subBoxLayoutID1
            orientation: 'horizontal'
            size_hint: (1, 0.1)
            Button:
                id: StartScreenButtonID
                text: "Play"
                on_release: kivy.app.App.get_running_app().toggleCV()
            Label:
                text: str(vidsliderID.value) #convert slider label to a time

FCVA_screen_manager: #remember to return a root widget
"""
        def build(self):
            self.title = self.shared_metadata_dictVAR["title"]
            build_app_from_kv = Builder.load_string(self.KV_string)
            button = Button(text="Test")
            inspector.create_inspector(Window, button)
            return build_app_from_kv

        def on_start(self):
            # start blitting. 1/30 always works because it will always blit the latest image from open_appliedcv subprocess, but kivy itself will be at 30 fps
            self.index = 0
            print("fps wtf", self.fps)
            from queue import Queue
            self.frameQ = Queue(maxsize=self.bufferlen*self.cvpartitions)
            self.internal_framecount = 0
            Clock.schedule_interval(self.blit_from_shared_memory, (1/self.fps))
            self.starttime = None

        def on_request_close(self, *args):
            Clock.unschedule(self.blit_from_shared_memory)
            print("#kivy subprocess closed END!", flush=True)

        def run(self):
            """Launches the app in standalone mode.
            reference:
            how to run kivy as a subprocess (so the main code can run neural networks like mediapipe without any delay)
            https://stackoverflow.com/questions/31458331/running-multiple-kivy-apps-at-same-time-that-communicate-with-each-other
            """
            self._run_prepare()
            from kivy.base import runTouchApp
            runTouchApp()
            self.shared_metadata_dictVAR["kivy_run_state"] = False

        def populate_texture(self, texture, buffervar):
            texture.blit_buffer(buffervar)
        
        def blit_from_shared_memory(self, *args):
            timeog = time.time()
            if "toggleCV" in self.shared_metadata_dictVAR and self.shared_globalindex_dictVAR["starttime"] != None:
                self.index = int((time.time() - self.starttime)/self.spf)
                if self.index < 0:
                    self.index = 0
                fprint("is cv subprocess keeping up?", self.index, self.shared_analyzedAKeycountVAR.values(),self.shared_analyzedBKeycountVAR.values(),self.shared_analyzedCKeycountVAR.values(),self.shared_analyzedDKeycountVAR.values())
                #cheat for rn, just get current frame:
                #know the current framenumber
                #get the right shareddict https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/#
                # https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
                # fprint("index in values?A",  self.index, self.shared_analyzedAKeycountVAR.values(), self.index in self.shared_analyzedAKeycountVAR.values())
                frame = None
                if self.index in self.shared_analyzedAKeycountVAR.values():
                    correctkey = list(self.shared_analyzedAKeycountVAR.keys())[list(self.shared_analyzedAKeycountVAR.values()).index(self.index)]
                    fprint("correctkey?", correctkey)
                    # if len(correctkey) > 0:
                    frameref = "frame" + correctkey.replace("key",'')
                    frame = self.shared_analyzedAVAR[frameref]
                
                # fprint("index in values?B",  self.index, self.shared_analyzedBKeycountVAR.values(), self.index in self.shared_analyzedBKeycountVAR.values())
                if self.index in self.shared_analyzedBKeycountVAR.values():
                    correctkey = list(self.shared_analyzedBKeycountVAR.keys())[list(self.shared_analyzedBKeycountVAR.values()).index(self.index)]
                    # fprint("correctkey?", correctkey)
                    # if len(correctkey) > 0:
                    frameref = "frame" + correctkey.replace("key",'')
                    timeax = time.time()
                    frame = self.shared_analyzedBVAR[frameref]
                    framesizeguy = frame
                    fprint("how long to load a frame from shared mem?", time.time()-timeax, "size?", sys.getsizeof(framesizeguy))

                # fprint("index in values?C",  self.index, self.shared_analyzedCKeycountVAR.values(), self.index in self.shared_analyzedCKeycountVAR.values())
                if self.index in self.shared_analyzedCKeycountVAR.values():
                    correctkey = list(self.shared_analyzedCKeycountVAR.keys())[list(self.shared_analyzedCKeycountVAR.values()).index(self.index)]
                    # fprint("correctkey?", correctkey)
                    # if len(correctkey) > 0:
                    frameref = "frame" + correctkey.replace("key",'')
                    frame = self.shared_analyzedCVAR[frameref]

                if self.index in self.shared_analyzedDKeycountVAR.values():
                    correctkey = list(self.shared_analyzedDKeycountVAR.keys())[list(self.shared_analyzedDKeycountVAR.values()).index(self.index)]
                    # fprint("correctkey?", correctkey)
                    # if len(correctkey) > 0:
                    frameref = "frame" + correctkey.replace("key",'')
                    frame = self.shared_analyzedDVAR[frameref]


                # https://stackoverflow.com/questions/43748991/how-to-check-if-a-variable-is-either-a-python-list-numpy-array-or-pandas-series
                try:
                    if frame != None:
                        frame = blosc2.unpack_array2(frame)
                        if isinstance(frame,np.ndarray):
                            buf = frame.tobytes()
                            frame = np.frombuffer(frame, np.uint8).copy().reshape(1080, 1920, 3)
                            
                            # complicated way of safely checking if a value may or may not exist, then get that value:
                            #quickly checked this, time is 0...
                            existence_check = [
                                frame.shape[x] for x in range(0, len(frame.shape)) if x == 2
                            ]
                            # only valid dimensions are if pixels are 3 (RGB) or 4 (RGBA, but u have to also set the colorfmt)
                            if [x for x in existence_check if x == 3 or x == 4] == []:
                                raise Exception(
                                    "check your numpy dimensions! should be (height, width, 3 for RGB/ 4 for RGBA): like  (1920,1080,3): ",
                                    frame.shape, frame
                                )
                            
                            # # check for existence of colorfmt in shared_metadata_dict, then if so, set colorfmt:
                            # formatoption = [
                            #     shared_metadata_dict[x]
                            #     for x in shared_metadata_dict.keys()
                            #     if x == "colorfmt"
                            # ]
                            # if len(formatoption) != 0:
                            #     self.colorfmtval = formatoption[0]
                            # else:
                            #     # default to bgr
                            #     self.colorfmtval = "bgr"

                            self.colorfmtval = "bgr"

                            # texture documentation: https://github.com/kivy/kivy/blob/master/kivy/graphics/texture.pyx
                            # blit to texture
                            # blit buffer example: https://stackoverflow.com/questions/61122285/kivy-camera-application-with-opencv-in-android-shows-black-screen

                            # I think creating a new texture is lagging the app, opencv reads the file faster than the video ends
                            # reference this, u need a reload observer: https://stackoverflow.com/questions/51546327/in-kivy-is-there-a-way-to-dynamically-change-the-shape-of-a-texture
                            # for later, if I need to clear a texture this is the reference: https://stackoverflow.com/questions/55099463/how-to-update-a-texture-from-array-in-kivy

                            # if hasattr(self, "texture1"):
                            #     print("texture size?", self.texture1.size[0] != frame.shape[1] and self.texture1.size[1] != frame.shape[0])
                            #     if (
                            #         self.texture1.size[0] != frame.shape[1]
                            #         and self.texture1.size[1] != frame.shape[0]
                            #     ):
                            #         print("texture size changed!", self.texture1.size)
                            #         self.texture1 = Texture.create(
                            #             size=(frame.shape[1], frame.shape[0]),
                            #             colorfmt=self.colorfmtval,
                            #         )
                            #         self.texture1.add_reload_observer(self.populate_texture)
                            #     else:
                            #         print("populating ok texture", flush= True)
                            #         self.populate_texture(self.texture1, buf)
                            # else:
                            #     print("notexture", flush= True)
                            #     self.texture1 = Texture.create(
                            #         size=(frame.shape[1], frame.shape[0]), colorfmt=self.colorfmtval
                            #     )
                            #     self.texture1.blit_buffer(
                            #         buf, colorfmt=self.colorfmtval, bufferfmt="ubyte"
                            #     )
                            #     self.texture1.add_reload_observer(self.populate_texture)

                            # print("blitting to texture index:", self.index)

                            self.texture1 = Texture.create(
                                size=(frame.shape[1], frame.shape[0]), colorfmt=self.colorfmtval
                            )
                            self.texture1.blit_buffer(
                                buf, colorfmt=self.colorfmtval, bufferfmt="ubyte"
                            )
                            App.get_running_app().root.get_screen("start_screen_name").ids[
                                "image_textureID"
                            ].texture = self.texture1
                    else:
                        if self.index != 0:
                            fprint("missed frame#", self.index)
                except Exception as e: 
                    print("blitting died!", e, flush=True)
                    import traceback
                    print("full exception", "".join(traceback.format_exception(*sys.exc_info())))
            self.newt = time.time()
            if hasattr(self, 'newt'):
                if self.newt - timeog > 0 and (1/(self.newt- timeog)) < 200:
                    print("blit fps?", 1/(self.newt- timeog))
                    pass
        
        def toggleCV(self, *args):
            # fprint("what are args, do I have widget?, nope, do the search strat", args)
            # fprint("id searching", )
            widgettext = App.get_running_app().root.get_screen('start_screen_name').ids['StartScreenButtonID'].text
            fprint("widgettext is?", widgettext)
            if "Play" in widgettext:
                App.get_running_app().root.get_screen('start_screen_name').ids['StartScreenButtonID'].text = "Pause"
                
                #check if you have been paused already:
                if "pausedtime" in self.shared_globalindex_dictVAR.keys() and isinstance(self.shared_globalindex_dictVAR["pausedtime"], float):
                    #start all subprocesses (hope it's fast enough):
                    subprocess_list = [x for x in self.shared_globalindex_dictVAR.keys() if "subprocess" in x]
                    for x in subprocess_list:
                        self.shared_globalindex_dictVAR[x] = True
                    #clear pausedtime and adjust starttime by elapsed time from last pause:
                    self.shared_globalindex_dictVAR["starttime"] = self.shared_globalindex_dictVAR["starttime"] + (time.time() - self.shared_globalindex_dictVAR["pausedtime"])
                    self.shared_globalindex_dictVAR["pausedtime"] = False
            else:
                App.get_running_app().root.get_screen('start_screen_name').ids['StartScreenButtonID'].text = "Play"
                
                self.shared_globalindex_dictVAR["pausedtime"] = time.time()
                fprint("#pause all subprocesses (hope it's fast enough):")
                subprocess_list = [x for x in self.shared_globalindex_dictVAR.keys() if "subprocess" in x]
                for x in subprocess_list:
                    self.shared_globalindex_dictVAR[x] = False
                
            if "toggleCV" not in self.shared_metadata_dictVAR.keys():
                self.shared_metadata_dictVAR["toggleCV"] = True
                if self.starttime == None:
                    #init starttime:
                    self.starttime = time.time() + 1
                    self.shared_globalindex_dictVAR["starttime"] = self.starttime
            else:
                # self.shared_metadata_dictVAR[
                #     "toggleCV"
                # ] = not self.shared_metadata_dictVAR["toggleCV"]
                #pop it to remove, that way I can make the time critical stuff faster:
                self.shared_metadata_dictVAR.pop("toggleCV")

    class FCVA_screen_manager(ScreenManager):
        pass

    class StartScreen(Screen):
        pass

    MainApp.shared_analysis_dictVAR = args[0]
    MainApp.shared_metadata_dictVAR = args[1]
    MainApp.fps = args[2]
    MainApp.shared_globalindex_dictVAR = args[3]
    MainApp.shared_analyzedAVAR = args[4]
    MainApp.shared_analyzedBVAR = args[5]
    MainApp.shared_analyzedCVAR = args[6]
    MainApp.shared_analyzedAKeycountVAR = args[7]
    MainApp.shared_analyzedBKeycountVAR = args[8]
    MainApp.shared_analyzedCKeycountVAR = args[9]
    MainApp.spf = args[10]
    MainApp.bufferlen = args[11]
    MainApp.cvpartitions = args[12]
    MainApp.framelength = args[13]
    MainApp.shared_analyzedDVAR = args[14]
    MainApp.shared_analyzedDKeycountVAR = args[15]

    MainApp().run()

def frameblock(*args):
    '''
    given partition #, instance, bufferlen, maxpartitions tells u the frames to get:

    ex: partitioning frames into A B C blocks (0-9 > A, 10-19> B, 20-29>C, etc) and buffer of 10
    then you know the partition: A (0) and instance: 0
        then you get (0>9)
    partition B (1) and instance 10 (so the 10th time this is done, index start at 0):
        then u get 110>120

    how to calculate the frameblock:
    know your bufferlen:
    shift the bufferlen by 2 things: the partition and the partition number
    partition number just adjusts your starting position by the number of bufferlengths you are from the start (so 0,1,2,3 * bufferlen)
    instance means how many full maxpartitions*bufferlen has already passed, so with maxpartition of 3 and bufferlen of 10, how many frames of 30 have already passed
    '''
    partitionnumber = args[0]
    instance = args[1]
    bufferlen = args[2]
    maxpartitions = args[3]
    print("frameblock args?", partitionnumber, instance)
    Ans = [x + bufferlen*maxpartitions*instance + partitionnumber*bufferlen for x in range(bufferlen)]
    return Ans

class open_cvpipeline_helper:
    pass

def open_cvpipeline(*args):
    try:
        shared_metadata_dict = args[0]
        appliedcv = args[1]
        shared_metadata_dict["mp_ready"] = True
        shared_analyzedVAR = args[2]
        shared_globalindex_dictVAR = args[3] #self.shared_globalindex_dictVAR["starttime"]
        shared_analyzedKeycountVAR = args[4]
        source = args[5]
        partitionnumber = args[6]
        instance = args[7]
        bufferlen = args[8]
        maxpartitions = args[9]
        fps = args[10]

        sourcecap = cv2.VideoCapture(source)
        internal_framecount = 0
        analyzedframecounter = 0
        instance_count = 0
        
        pid = os.getpid()
        shared_globalindex_dictVAR["subprocess" + str(pid)] = True

        from queue import Queue
        raw_queue = Queue(maxsize=bufferlen)
        raw_queueKEYS = Queue(maxsize=bufferlen)
        analyzed_queue = Queue(maxsize=bufferlen)
        analyzed_queueKEYS = Queue(maxsize=bufferlen)
        
        open_cvpipeline_helper_instance = open_cvpipeline_helper()
        open_cvpipeline_helper_instance.resultsq = Queue(maxsize=bufferlen)

        #init mediapipe here so it spawns the right amt of processes
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        with open('I:\CODING\FastCVApp\FastCVApp\examples\creativecommonsmedia\pose_landmarker_full.task', 'rb') as f:
                    modelbytes = f.read()
                    base_options = python.BaseOptions(model_asset_buffer=modelbytes)
                    VisionRunningMode = mp.tasks.vision.RunningMode
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=VisionRunningMode.VIDEO,
                        #these were old settings, maybe it's too strict and not giving me poses
                        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5
                        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

        while True:
            if "kivy_run_state" in shared_metadata_dict:
                if shared_metadata_dict["kivy_run_state"] == False:
                    print("exiting open_appliedcv", os.getpid(), flush=True)
                    break
                '''
                NEW PLAN:
                Init shared dicts at the beginning instead of checking every while loop
                
                use 3 subprocesses(A,B,C) to use opencv to get frames from 1 file simultaneously (pray it works and there's no file hold...)
                then for each subprocesses, request 10 frames (0-9 > A, 10-19> B, 20-39>C, etc)
                2 queues, 1 naked frame, 1 analyzed frame that is written to sharedmem for kivy to see
                2 dicts:
                rawqueue
                analyzedqueue

                LOOP:
                    3 actions: 
                    Read
                        request the RIGHT 10 frames (0-10 or 11-20 or 21-30)
                        Load raw frames only if analyze queue is empty (this implicitly checks for time, keeps frames loaded, and stops u from loading too much)
                    Analyze
                        Analyze all the time (if analyze queue is empty and there is a framequeue)
                    Write
                        Write to shared dict if init OR frames are old
                '''
                #make sure things have started AND this processess is not stopped:
                if "starttime" in shared_globalindex_dictVAR and shared_globalindex_dictVAR["subprocess" + str(pid)]:

                    initial_time = time.time()
                    future_time = shared_globalindex_dictVAR["starttime"] + ((1/fps)*internal_framecount)
                    current_framenumber = int((time.time() - shared_globalindex_dictVAR["starttime"])/(1/fps))
                    fprint("frame advantage START????", os.getpid(), internal_framecount, current_framenumber, future_time-time.time(), time.time())
                    
                    if raw_queue.qsize() == 0:
                        #get the right framecount:
                        framelist = frameblock(partitionnumber,instance_count,bufferlen,maxpartitions)
                        fprint("says true for some reason?", shared_globalindex_dictVAR["subprocess" + str(pid)])
                        instance_count += 1
                        for x in range(bufferlen*maxpartitions):
                            (ret, framedata) = sourcecap.read()
                            #compare internal framecount to see if it's a frame that this subprocess is supposed to analyze
                            if ret and internal_framecount in framelist:
                                raw_queue.put(framedata)
                                raw_queueKEYS.put(framelist[x % bufferlen])
                            internal_framecount += 1
                            # fprint("ret, queue, keys, internal",ret, type(framedata), framelist[x % bufferlen], internal_framecount)
                            current_framenumber = int((time.time() - shared_globalindex_dictVAR["starttime"])/(1/fps))
                            if not ret and current_framenumber > internal_framecount+fps: #if ret is false, and we passed EOS (add 1 second (fps amount of frames) from internal_framecount AKA current_framenumber > internal_framecount + fps)
                                shared_globalindex_dictVAR["subprocess" + str(pid)] = ret #say so in PID and wait for another process to reset it
                                fprint("PID STOPPED", pid, internal_framecount)
                    
                    if raw_queue.qsize() > 0 and analyzed_queue.qsize() == 0:
                        #give the queue to the cv func
                        #cv func returns a queue of frames
                        rtime = time.time()
                        resultqueue = appliedcv(open_cvpipeline_helper_instance, raw_queue, shared_globalindex_dictVAR, shared_metadata_dict, bufferlen, landmarker)
                        fprint("resultqueue timing", time.time() - rtime, os.getpid(), time.time())
                        fprint("#then get from the analyzed queue and apply blosc2", resultqueue.qsize())
                        current_framenumber = int((time.time() - shared_globalindex_dictVAR["starttime"])/(1/fps))

                        #figure out future time
                        future_time = shared_globalindex_dictVAR["starttime"] + ((1/fps)*internal_framecount)

                        # fprint("frame advantage????", os.getpid(), internal_framecount, current_framenumber, future_time-time.time())
                        for x in range(resultqueue.qsize()):
                            result_compressed = blosc2.pack_array2(resultqueue.get())
                            analyzed_queue.put(result_compressed)
                            analyzed_queueKEYS.put(raw_queueKEYS.get())
                            # fprint("x in range?", x)


                        # #analyze all the frames and write to sharedmem:
                        # for x in range(raw_queue.qsize()):
                        #     result = appliedcv(
                        #                 raw_queue.get(),
                        #             )
                        #     #compress the numpy array with blosc so that reading is not as bad of a bottleneck
                        #     result_compressed = blosc2.pack_array2(result)
                        #     analyzed_queue.put(result_compressed)
                        #     analyzed_queueKEYS.put(raw_queueKEYS.get())
                    
                    current_framenumber = int((time.time() - shared_globalindex_dictVAR["starttime"])/(1/fps))
                    if analyzed_queue.qsize() == bufferlen and max(shared_analyzedKeycountVAR.values()) < current_framenumber:
                        dictwritetime = time.time()
                        for x in range(bufferlen):
                            shared_analyzedVAR['frame'+str(x)] = analyzed_queue.get()
                            shared_analyzedKeycountVAR['key'+str(x)] = analyzed_queueKEYS.get()
                        fprint("dictwritetime", time.time()-dictwritetime, os.getpid(), time.time())
                    fprint("frame advantage END????", os.getpid(), internal_framecount, current_framenumber, future_time-time.time(), time.time(), "total time?", time.time() - initial_time)

                    # print("what are analyzed keys?", shared_analyzedKeycountVAR.values(), flush = True)
    except Exception as e:
        print("open_appliedcv died!", e)
        import traceback
        print("full exception", "".join(traceback.format_exception(*sys.exc_info())))

class FCVA:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.appliedcv = None

    def run(self):
        if __name__ == "FastCVApp":
            import multiprocessing as FCVA_mp

            # this is so that only 1 window is run when packaging with pyinstaller
            FCVA_mp.freeze_support()
            shared_mem_manager = FCVA_mp.Manager()
            # shared_analysis_dict holds the actual frames
            shared_analysis_dict = shared_mem_manager.dict()
            # shared_metadata_dict holds keys about run states so things don't error by reading something that doesn't exist
            shared_metadata_dict = shared_mem_manager.dict()
            # FEAR OF USING SHARED METADATA DICT TOO MUCH: too many processes lock up the memory too much....
            # 2nd shared metadata dict: shared global index, knows: current frame, paused time, idk what else...
            shared_globalindex_dict = shared_mem_manager.dict()
            shared_globalindex_dict["curframe"] = 0


            # shared_poolmeta_dict = shared_mem_manager.dict()
            # analyze_pool_count = 3
            # for x in range(analyze_pool_count):
            #     shared_poolmeta_dict[x] = 

            shared_analyzedA = shared_mem_manager.dict()
            shared_analyzedAKeycount = shared_mem_manager.dict()
            shared_analyzedB = shared_mem_manager.dict()
            shared_analyzedBKeycount = shared_mem_manager.dict()
            shared_analyzedC = shared_mem_manager.dict()
            shared_analyzedCKeycount = shared_mem_manager.dict()
            shared_analyzedD = shared_mem_manager.dict()
            shared_analyzedDKeycount = shared_mem_manager.dict()
            

            # set metadata kivy_run_state to true so cv subprocess will run and not get an error by reading uninstantiated shared memory.
            shared_metadata_dict["kivy_run_state"] = True

            # reference: https://stackoverflow.com/questions/8220108/how-do-i-check-the-operating-system-in-python
            from sys import platform

            if platform == "linux" or platform == "linux2":
                # linux
                pass
            elif platform == "darwin_old":
                # OS X, need to change filepath so pyinstaller exe will work
                mac_path = (
                    os.path.sep.join(sys.argv[0].split(os.path.sep)[:-1]) + os.path.sep
                )
                print("mac option", mac_path)
                print("what is self source then?", self.source)
                # vanity code so example works from main file or from examples folder
                if "examples" in mac_path:
                    mac_source = self.source
                else:
                    mac_source = mac_path + self.source

                # check if file exists in dir, if not then check tmp folder, if nothing, raise error:
                # reference: https://stackoverflow.com/questions/54837659/python-pyinstaller-on-mac-current-directory-problem
                if os.path.isfile(mac_source):
                    print("file exists in dir ", mac_source)
                    self.source = mac_source
                elif not os.path.isfile(mac_source):
                    print(
                        "File not in .dmg directory, location failed isfile check, checking tmp dir: ",
                        mac_source,
                    )

                # checking tempfolder
                if hasattr(sys, "_MEIPASS"):
                    # if file is frozen by pyinstaller add the MEIPASS folder to path:
                    sys.path.append(sys._MEIPASS)
                    tempsource = sys._MEIPASS + os.sep + self.source

                    if os.path.isfile(tempsource):
                        self.source = tempsource
                    elif not os.path.isfile(tempsource):
                        raise Exception(
                            "Source failed isfile check: " + str(tempsource)
                        )

            elif platform == "win32" or platform == "darwin":
                # Windows...
                # check current directory, then check tmpfolder, then complain:

                # if you're in examples folder, path is gonna be wrong, so fix it:
                dirlist = os.getcwd().split(os.path.sep)
                if "examples" in dirlist[-1]:
                    # pathjoin is weird: https://stackoverflow.com/questions/2422798/python-os-path-join-on-windows
                    dirlist_source = (
                        dirlist[0]
                        + os.path.sep
                        + os.path.join(*dirlist[1 : len(dirlist) - 1])
                        + os.path.sep
                        + self.source
                    )
                    if not os.path.isfile(dirlist_source):
                        print("not a playable file: ??", dirlist_source)
                    else:
                        self.source = dirlist_source
                # NOW check current directory:
                elif os.path.isfile(self.source):
                    print("file loaded:", os.getcwd() + os.sep + self.source)
                elif not os.path.isfile(self.source):
                    print(
                        "Source failed isfile check for current directory: "
                        + str(os.path.isfile(self.source))
                        + ". Checking location: "
                        + str(os.path.join(os.getcwd(), self.source))
                        + " Checking tmpdir next:"
                    )

                # print("#check sys attr:", hasattr(sys, '_MEIPASS'))
                if hasattr(sys, "_MEIPASS"):
                    # if file is frozen by pyinstaller add the MEIPASS folder to path:
                    sys.path.append(sys._MEIPASS)
                    tempsource = sys._MEIPASS + os.sep + self.source

                    if os.path.isfile(tempsource):
                        self.source = tempsource
                    # checked everything, now complain:
                    elif not os.path.isfile(tempsource):
                        raise Exception(
                            "Source failed isfile check: " + str(tempsource)
                        )

            # read just to get the fps
            video = cv2.VideoCapture(self.source)
            self.fps = video.get(cv2.CAP_PROP_FPS)
            #opencv is accurately guessing, read through everything for accuracy:
            # https://stackoverflow.com/questions/31472155/python-opencv-cv2-cv-cv-cap-prop-frame-count-get-wrong-numbers
            # self.length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.length = 0
            while True: 
                ret, framevar = video.read()
                if not ret:
                    break
                self.length += 1
            video.release()

            bufferlen = 10
            cvpartitions = 4
            #init shared dicts:
            for x in range(bufferlen):
                shared_analyzedAKeycount["key" + str(x)] = -1
                shared_analyzedA["frame" + str(x)] = -1

                shared_analyzedBKeycount["key" + str(x)] = -1
                shared_analyzedB["frame" + str(x)] = -1

                shared_analyzedCKeycount["key" + str(x)] = -1
                shared_analyzedC["frame" + str(x)] = -1

                shared_analyzedDKeycount["key" + str(x)] = -1
                shared_analyzedD["frame" + str(x)] = -1
            
            #sanity checks
            if not hasattr(self, "fps"):
                # default to 30fps, else set blit buffer speed to 1/30 sec
                self.fps = 1 / 30
            if not hasattr(self, "title"):
                shared_metadata_dict[
                    "title"
                ] = "Fast CV App Example v0.1.0 by Pengindoramu"
            else:
                shared_metadata_dict["title"] = self.title
            if hasattr(self, "colorfmt"):
                shared_metadata_dict["colorfmt"] = self.colorfmt
            if hasattr(self, "kvstring"):
                shared_metadata_dict["kvstring"] = self.kvstring
            if self.appliedcv == None:
                print(
                    "FCVA.appliedcv is currently None. Not starting the CV subprocess."
                )
            
            #start the subprocesses
            cv_subprocessA = FCVA_mp.Process(
                    target=open_cvpipeline,
                    args=(
                        shared_metadata_dict,
                        self.appliedcv,
                        shared_analyzedA,
                        shared_globalindex_dict,
                        shared_analyzedAKeycount,
                        self.source,
                        0, #partition #, starts at 0
                        0, #instance of the block of relevant frames
                        bufferlen, #bufferlen AKA how long the internal queues should be
                        cvpartitions, #max # of partitions/subprocesses that divide up the video sequence
                        self.fps,
                    ),
                )
            cv_subprocessA.start()

            cv_subprocessB = FCVA_mp.Process(
                    target=open_cvpipeline,
                    args=(
                        shared_metadata_dict,
                        self.appliedcv,
                        shared_analyzedB,
                        shared_globalindex_dict,
                        shared_analyzedBKeycount,
                        self.source,
                        1, #partition #, starts at 0
                        0, #instance of the block of relevant frames
                        bufferlen, #bufferlen AKA how long the internal queues should be
                        cvpartitions, #max # of partitions/subprocesses that divide up the video sequence
                        self.fps,
                    ),
                )
            cv_subprocessB.start()

            cv_subprocessC = FCVA_mp.Process(
                    target=open_cvpipeline,
                    args=(
                        shared_metadata_dict,
                        self.appliedcv,
                        shared_analyzedC,
                        shared_globalindex_dict,
                        shared_analyzedCKeycount,
                        self.source,
                        2, #partition #, starts at 0
                        0, #instance of the block of relevant frames
                        bufferlen, #bufferlen AKA how long the internal queues should be
                        cvpartitions, #max # of partitions/subprocesses that divide up the video sequence
                        self.fps,
                    ),
                )
            cv_subprocessC.start()

            cv_subprocessD = FCVA_mp.Process(
                    target=open_cvpipeline,
                    args=(
                        shared_metadata_dict,
                        self.appliedcv,
                        shared_analyzedD,
                        shared_globalindex_dict,
                        shared_analyzedDKeycount,
                        self.source,
                        2, #partition #, starts at 0
                        0, #instance of the block of relevant frames
                        bufferlen, #bufferlen AKA how long the internal queues should be
                        cvpartitions, #max # of partitions/subprocesses that divide up the video sequence
                        self.fps,
                    ),
                )
            cv_subprocessD.start()

            kivy_subprocess = FCVA_mp.Process(
                target=open_kivy,
                args=(shared_analysis_dict, shared_metadata_dict, self.fps, shared_globalindex_dict, shared_analyzedA, shared_analyzedB, shared_analyzedC,shared_analyzedAKeycount,shared_analyzedBKeycount, shared_analyzedCKeycount, (1/self.fps), bufferlen,cvpartitions, self.length, shared_analyzedD, shared_analyzedDKeycount)
                # 
            )
            kivy_subprocess.start()

            # this try except block holds the main process open so the subprocesses aren't cleared when the main process exits early.
            while "kivy_run_state" in shared_metadata_dict.keys():
                if shared_metadata_dict["kivy_run_state"] == False:
                    # when the while block is done, close all the subprocesses using .join to gracefully exit. also make sure opencv releases the video.
                    cv_subprocessA.join()
                    cv_subprocessB.join()
                    cv_subprocessC.join()
                    cv_subprocessD.join()
                    kivy_subprocess.join()
                    break
                try:
                    pass
                except Exception as e:
                    print(
                        "Error in run, make sure stream is set. Example: app.source = 0 (so opencv will open videocapture 0)",
                        e,
                    )