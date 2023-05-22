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
                self.KV_string = """
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
        Image:
            id: image_textureID
        Button:
            id: StartScreenButton
            text: "Start analyzing!"
            on_release: kivy.app.App.get_running_app().toggleCV()

FCVA_screen_manager: #remember to return a root widget
"""
        def build(self):
            self.title = self.shared_metadata_dictVAR["title"]
            build_app_from_kv = Builder.load_string(self.KV_string)
            return build_app_from_kv

        def on_start(self):
            # start blitting. 1/30 always works because it will always blit the latest image from open_appliedcv subprocess, but kivy itself will be at 30 fps
            self.index = 0
            print("fps wtf", self.fps)
            from queue import Queue
            self.frameQ = Queue(maxsize=self.bufferlen*self.cvpartitions)
            self.internal_framecount = 0
            Clock.schedule_interval(self.blit_from_shared_memory, (1/self.fps))
            # Clock.schedule_interval(self.blit_from_shared_memory, 1/60)
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
            if "toggleCV" in self.shared_metadata_dictVAR and self.shared_globalindexVAR["starttime"] != None:
                self.index = int((time.time() - self.starttime)/self.spf)
                if self.index < 0:
                    self.index = 0
                
                #figure out where 
                # self.index is the current realtime frame btw
                
                # initialize the framequeue onstart
                # self.frameQ
                
                # load as much as you can:
                # check if there's enough space for 1 bufferlen
                
                fprint("is cv subprocess keeping up?", self.index, self.shared_analyzedAKeycountVAR.values(),self.shared_analyzedBKeycountVAR.values(),self.shared_analyzedCKeycountVAR.values())
                #cheat for rn, just get current frame:
                #know the current framenumber
                #get the right shareddict https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/#
                # https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
                # fprint("index in values?A",  self.index, self.shared_analyzedAKeycountVAR.values(), self.index in self.shared_analyzedAKeycountVAR.values())
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
                    frame = self.shared_analyzedBVAR[frameref]

                # fprint("index in values?C",  self.index, self.shared_analyzedCKeycountVAR.values(), self.index in self.shared_analyzedCKeycountVAR.values())
                if self.index in self.shared_analyzedCKeycountVAR.values():
                    correctkey = list(self.shared_analyzedCKeycountVAR.keys())[list(self.shared_analyzedCKeycountVAR.values()).index(self.index)]
                    # fprint("correctkey?", correctkey)
                    # if len(correctkey) > 0:
                    frameref = "frame" + correctkey.replace("key",'')
                    frame = self.shared_analyzedCVAR[frameref]


                #display 
                # correctkey
                
                '''
                if self.frameQ.qsize() < self.bufferlen*(self.cvpartitions - 1) :
                    fprint("checking keys dict values", self.shared_analyzedAKeycountVAR.values(), self.shared_analyzedBKeycountVAR.values(), self.shared_analyzedCKeycountVAR.values(), self.index)
                    #PLAN:
                    #copy dictionary and time it
                    #https://www.programiz.com/python-programming/methods/dictionary/copy
                    # https://stackoverflow.com/questions/2465921/how-to-copy-a-dictionary-and-only-edit-the-copy
                    timeog1 = time.time()
                    # newdict = self.shared_analyzedAVAR.copy()
                    # newdict = blosc2.unpack(self.shared_analyzedAVAR[self.shared_analyzedAVAR.keys()[0]])
                    newdict = self.shared_analyzedAVAR[self.shared_analyzedAVAR.keys()[0]]
                    if newdict != -1:
                        newdict = blosc2.unpack(newdict)
                    # newdict2 = self.shared_analyzedAVAR[self.shared_analyzedAVAR.keys()[1]]
                    # fprint("keys???", self.shared_analyzedAVAR.values())
                    timeog2 = time.time()
                    fprint("how long to load?", timeog2 - timeog1, sys.getsizeof(newdict))
                '''
                

                    #then think...
                #     #read in only 1 block sequence so there's no stutter
                #     #given self.internal_framecount, what is the next block to read in? -> 0>9... at 9, read 9>19, etc...
                #     if find the next framekeys in the list of all the keys AND framekeys are the entire block (so we know analysis is all done): 
                #         read it in sequence and add to frameQ
                #         +1 on internal read data
                #         self.internal_framecount += 1
                    


                # how to load?
                # NOT CORRECT BECAUSE it's a waste of time to read correct block, it's better to read as much as you can and stuff into a queue imo
                #     figure out which sharedmem self index is in
                #     load that block of frames,
                # if possible load the future block of frames as well

                # else: pull from queue and display it:
                # if queuesize is empty, say so


                #manually code this for now:
                # # if self.index %3 == 0 or self.index %3 == 1 or self.index %3 == 2:
                # if self.index %3 == 0:
                #     # print("key vs me", self.shared_speedtestAVAR.keys(), type(self.shared_speedtestAVAR.keys()[0]), self.index, self.index %2, type(self.index%2) )
                    
                #     #now u have to search for self.index in shared_analyzedAVAR.keys for the right key:
                #     # sharedanalyzedkeysA = self.shared_analyzedAVAR.keys()
                #     keyref = [x for x in self.shared_analyzedAKeycountVAR.keys() if 'key' in x and self.shared_analyzedAKeycountVAR[x] == self.index]
                #     if keyref == []:
                #         print("keyfauiledA", self.index, [self.shared_analyzedAKeycountVAR[x] for x in self.shared_analyzedAKeycountVAR.keys() if 'key' in x] , flush = True)
                #         # print("keyref fail! A,",self.index, keyref, self.shared_analyzedAVAR.keys(),[self.shared_analyzedAVAR[x] for x in self.shared_analyzedAVAR.keys() if isinstance(self.shared_analyzedAVAR[x],int)],  flush = True)
                #         pass
                #     else:
                #         frameref = "frame" + keyref[0].replace("key",'')
                #         # print("frame passed?A", frameref, self.index, self.shared_analyzedAVAR[keyref[0]], self.index == self.shared_analyzedAVAR[keyref[0]], self.shared_analyzedAVAR.keys(), flush = True)
                #         frame = self.shared_analyzedAVAR[frameref]
                #     # frame = self.shared_analyzedAVAR[self.index]
                #     # self.shared_analyzedAVAR.pop(self.index)
                #     # #delete all the keys < our index:
                #     # [self.shared_analyzedAVAR.pop(x) for x in self.shared_analyzedAVAR.keys() if x < self.index]
                #     # print("why is it getting bigger? A(reading function isn't throttled....)", self.index, self.shared_analyzedAVAR.keys())
                # if self.index %3 == 1:
                #     # print("key vs me", self.shared_speedtestBVAR.keys(), type(self.shared_speedtestBVAR.keys()[0]), self.index, self.index %2, type(self.index%2) )
                #     # sharedanalyzedB = self.shared_analyzedBVAR.keys()
                #     keyref = [x for x in self.shared_analyzedBKeycountVAR.keys() if 'key' in x and self.shared_analyzedBKeycountVAR[x] == self.index]
                #     if keyref == []:
                #         # print("keyref fail! B,",self.index, keyref, self.shared_analyzedBVAR.keys(),[self.shared_analyzedBVAR[x] for x in self.shared_analyzedBVAR.keys() if isinstance(self.shared_analyzedBVAR[x],int)], flush = True)
                #         # print("keyfauiledB", self.index, [self.shared_analyzedBVAR[x] for x in self.shared_analyzedBVAR.keys() if 'key' in x] , flush = True)
                #         pass
                #     else:
                #         frameref = "frame" + keyref[0].replace("key",'')
                #         # print("frame passed?B", frameref, self.index, self.shared_analyzedBVAR[keyref[0]], self.index == self.shared_analyzedBVAR[keyref[0]], self.shared_analyzedBVAR.keys(), flush = True)
                #         frame = self.shared_analyzedBVAR[frameref]
                #     # frame = self.shared_analyzedBVAR[self.index]
                #     # self.shared_analyzedBVAR.pop(self.index)
                #     # [self.shared_analyzedBVAR.pop(x) for x in self.shared_analyzedBVAR.keys() if x < self.index]
                #     # print("why is it getting bigger? B(reading function isn't throttled....)", self.index, self.shared_analyzedBVAR.keys())
                # if self.index %3 == 2:
                #     # print("key vs me", self.shared_speedtestCVAR.keys(), type(self.shared_speedtestCVAR.keys()[0]), self.index, self.index %2, type(self.index%2) )
                #     # sharedanalyzedkeysC = self.shared_analyzedCVAR.keys()
                #     keyref = [x for x in self.shared_analyzedCKeycountVAR.keys() if 'key' in x and self.shared_analyzedCKeycountVAR[x] == self.index]
                #     if keyref == []:
                #         # print("keyref fail! C,",self.index, keyref, self.shared_analyzedCVAR.keys(),[self.shared_analyzedCVAR[x] for x in self.shared_analyzedCVAR.keys() if isinstance(self.shared_analyzedAVAR[x],int)],flush = True)
                #         # print("keyfauiledC", self.index, [self.shared_analyzedCVAR[x] for x in self.shared_analyzedCVAR.keys() if 'key' in x] , flush = True)
                #         pass
                #     else:
                #         frameref = "frame" + keyref[0].replace("key",'')
                #         # print("frame passed?C", frameref, self.index, self.shared_analyzedCVAR[keyref[0]], self.index == self.shared_analyzedCVAR[keyref[0]], self.shared_analyzedCVAR.keys(), flush = True)
                #         frame = self.shared_analyzedCVAR[frameref]
                #     # frame = self.shared_analyzedCVAR[self.index]
                #     # self.shared_analyzedCVAR.pop(self.index)
                #     # [self.shared_analyzedCVAR.pop(x) for x in self.shared_analyzedCVAR.keys() if x < self.index]
                #     # print("why is it getting bigger? C(reading function isn't throttled....)", self.index, self.shared_analyzedCVAR.keys())
                
                # self.newt = time.time()

                # #this is def slow...
                # # try: 
                # #     frame
                # # except:
                # #     frame = None
                
                # # https://stackoverflow.com/questions/43748991/how-to-check-if-a-variable-is-either-a-python-list-numpy-array-or-pandas-series
                # # if not isinstance(frame,np.ndarray):

                # # # dummyinfo for speed testing
                # # dummyframe = np.full((1920,1080, 3), [180, 180, 180], dtype=np.uint8)
                # # dummyframe = dummyframe.tobytes()
                # # frame = dummyframe
                # # keyref = [[]]
                
                # if frame is None:
                # if keyref == []:
                #     # print("frame ded")
                #     pass
                # else:
                try:
                    #frame is already in bytes, just reshape it then reset to bytes again
                    # frame = blosc2.unpack(frame)
                    frame = blosc2.unpack_array2(frame)
                    buf = frame.tobytes()
                    # buf = frame.tobytes()
                    frame = np.frombuffer(frame, np.uint8).copy().reshape(1080, 1920, 3)
                    #TURN THIS BACK ON
                    '''
                    # complicated way of safely checking if a value may or may not exist, then get that value:
                    existence_check = [
                        frame.shape[x] for x in range(0, len(frame.shape)) if x == 2
                    ]
                    # only valid dimensions are if pixels are 3 (RGB) or 4 (RGBA, but u have to also set the colorfmt)
                    if [x for x in existence_check if x == 3 or x == 4] == []:
                        raise Exception(
                            "check your numpy dimensions! should be height x width x 3/4: like  (1920,1080,3): ",
                            frame.shape, frame
                        )
                    '''
                    # buf = frame.tobytes()
                    
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
            if "toggleCV" not in self.shared_metadata_dictVAR.keys():
                self.shared_metadata_dictVAR["toggleCV"] = True
                if self.starttime == None:
                    #init starttime:
                    self.starttime = time.time() + 1
                    self.shared_globalindexVAR["starttime"] = self.starttime
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
    MainApp.shared_globalindexVAR = args[3]
    MainApp.shared_analyzedAVAR = args[4]
    MainApp.shared_analyzedBVAR = args[5]
    MainApp.shared_analyzedCVAR = args[6]
    MainApp.shared_analyzedAKeycountVAR = args[7]
    MainApp.shared_analyzedBKeycountVAR = args[8]
    MainApp.shared_analyzedCKeycountVAR = args[9]
    MainApp.spf = args[10]
    MainApp.bufferlen = args[11]
    MainApp.cvpartitions = args[12]

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

def open_cvpipeline(*args):
    try:
        shared_metadata_dict = args[0]
        appliedcv = args[1]
        shared_metadata_dict["mp_ready"] = True
        shared_analyzedVAR = args[2]
        shared_globalindexVAR = args[3] #self.shared_globalindexVAR["starttime"]
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

        from queue import Queue
        raw_queue = Queue(maxsize=bufferlen)
        raw_queueKEYS = Queue(maxsize=bufferlen)
        analyzed_queue = Queue(maxsize=bufferlen)
        analyzed_queueKEYS = Queue(maxsize=bufferlen)

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
                #make sure things have started:
                if "starttime" in shared_globalindexVAR:
                    if raw_queue.qsize() == 0:
                        #get the right framecount:
                        framelist = frameblock(partitionnumber,instance_count,bufferlen,maxpartitions)
                        instance_count += 1
                        for x in range(bufferlen*maxpartitions):
                            (ret, framedata) = sourcecap.read()
                            #compare internal framecount to see if it's a frame that this subprocess is supposed to analyze
                            if ret and internal_framecount in framelist:
                                raw_queue.put(framedata)
                                raw_queueKEYS.put(framelist[x % bufferlen])
                            internal_framecount += 1
                    
                    if raw_queue.qsize() > 0 and analyzed_queue.qsize() == 0:
                        #analyze all the frames and write to sharedmem:
                        for x in range(raw_queue.qsize()):
                            result = appliedcv(
                                        raw_queue.get(),
                                    )
                            #compress the numpy array with blosc so that reading is not as bad of a bottleneck
                            result_compressed = blosc2.pack_array2(result)
                            analyzed_queue.put(result_compressed)
                            analyzed_queueKEYS.put(raw_queueKEYS.get())
                    
                    current_framenumber = int((time.time() - shared_globalindexVAR["starttime"])/(1/fps))
                    if analyzed_queue.qsize() == bufferlen and max(shared_analyzedKeycountVAR.values()) < current_framenumber:
                        for x in range(bufferlen):
                            shared_analyzedVAR['frame'+str(x)] = analyzed_queue.get()
                            shared_analyzedKeycountVAR['key'+str(x)] = analyzed_queueKEYS.get()

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
            # shared_speedtest = shared_mem_manager.dict() #split off into A, B, C
            
            # shared_poolmeta_dict = shared_mem_manager.dict()
            # analyze_pool_count = 3
            # for x in range(analyze_pool_count):
            #     shared_poolmeta_dict[x] = 
            
            shared_speedtestA = shared_mem_manager.dict()
            shared_speedtestAKeycount = shared_mem_manager.dict()
            shared_speedtestB = shared_mem_manager.dict()
            shared_speedtestBKeycount = shared_mem_manager.dict()
            shared_speedtestC = shared_mem_manager.dict()
            shared_speedtestCKeycount = shared_mem_manager.dict()

            shared_analyzedA = shared_mem_manager.dict()
            shared_analyzedAKeycount = shared_mem_manager.dict()
            shared_analyzedB = shared_mem_manager.dict()
            shared_analyzedBKeycount = shared_mem_manager.dict()
            shared_analyzedC = shared_mem_manager.dict()
            shared_analyzedCKeycount = shared_mem_manager.dict()
            
            shared_globalindex = shared_mem_manager.dict()
            shared_globalindex["curframe"] = 0

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
            # print("args ok?", shared_metadata_dict, fps, self.source, os.path.isfile(self.source))

            # read_subprocessTEST = FCVA_mp.Process(
            #     target=open_mediaTEST, args=(shared_metadata_dict, self.fps, self.source, shared_speedtestA, shared_speedtestB, shared_speedtestC, shared_globalindex, shared_analyzedA, shared_analyzedB, shared_analyzedC,shared_speedtestAKeycount,shared_speedtestBKeycount,shared_speedtestCKeycount)
            # )
            # read_subprocessTEST.start()

            bufferlen = 10
            cvpartitions = 3
            #init shared dicts:
            for x in range(bufferlen):
                shared_analyzedAKeycount["key" + str(x)] = -1
                shared_analyzedA["frame" + str(x)] = -1

                shared_analyzedBKeycount["key" + str(x)] = -1
                shared_analyzedB["frame" + str(x)] = -1

                shared_analyzedCKeycount["key" + str(x)] = -1
                shared_analyzedC["frame" + str(x)] = -1
            

            cv_subprocessA = FCVA_mp.Process(
                    target=open_cvpipeline,
                    args=(
                        shared_metadata_dict,
                        self.appliedcv,
                        shared_analyzedA,
                        shared_globalindex,
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
                        shared_globalindex,
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
                        shared_globalindex,
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

            # shared_globalindex["starttime"] = time.time() + 2
            kivy_subprocess = FCVA_mp.Process(
                target=open_kivy,
                args=(shared_analysis_dict, shared_metadata_dict, self.fps, shared_globalindex, shared_analyzedA, shared_analyzedB, shared_analyzedC,shared_analyzedAKeycount,shared_analyzedBKeycount,shared_analyzedCKeycount, (1/self.fps), bufferlen,cvpartitions)
            )
            
            
            kivy_subprocess.start()
            


            '''#TURN THIS ON
            elif self.appliedcv == None:
                print(
                    "FCVA.appliedcv is currently None. Not starting the CV subprocess."
                )
            else:
                print("FCVA.appliedcv block failed")

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

            # shared_globalindex["starttime"] = time.time() + 2
            kivy_subprocess = FCVA_mp.Process(
                target=open_kivy,
                args=(shared_analysis_dict, shared_metadata_dict, self.fps, shared_globalindex, shared_analyzedA, shared_analyzedB, shared_analyzedC)
            )
            #old args: args=(shared_analysis_dict, shared_metadata_dict, self.fps, shared_speedtestA,shared_speedtestB,shared_speedtestC, shared_globalindex, shared_analyzedA, shared_analyzedB, shared_analyzedC)

            #dummytesting
            

            kivy_subprocess.start()
            '''

            # this try except block holds the main process open so the subprocesses aren't cleared when the main process exits early.
            while "kivy_run_state" in shared_metadata_dict.keys():
                if shared_metadata_dict["kivy_run_state"] == False:
                    # when the while block is done, close all the subprocesses using .join to gracefully exit. also make sure opencv releases the video.
                    read_subprocess.join()
                    cv_subprocess.join()
                    cv_subprocessB.join()
                    cv_subprocessC.join()
                    video.release()
                    kivy_subprocess.join()
                    break
                try:
                    pass
                except Exception as e:
                    print(
                        "Error in run, make sure stream is set. Example: app.source = 0 (so opencv will open videocapture 0)",
                        e,
                    )
