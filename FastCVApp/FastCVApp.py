# so that main and subprocesses have access to this since it's not under if __name__ is main
import cv2
import time
import os, sys
import numpy as np
from FCVAutils import fprint
#blosc uses multiprocessing, call it after freeze support so exe doesn't hang
#https://github.com/pyinstaller/pyinstaller/issues/7470#issuecomment-1448502333
#I immediately call multiprocessing.freeze_support() in example_mediapipe but it's not good for abstraction, think about it
import blosc2

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
    # print("frameblock args?", partitionnumber, instance)
    Ans = [x + bufferlen*maxpartitions*instance + partitionnumber*bufferlen for x in range(bufferlen)]
    return Ans

def int_to_partition(*args):
    '''
    args: 
        int u want to test
        bufferlen
        maxpartitions
    returns:
        partition# that contains this int any the frameblock

    78 > 70>80 correct?
    then u want NOT the mod, but the whole #:

    78 % (bufferlen) = 8
    78 - (78 % bufferlen) = 70
    (78 - (78 % bufferlen))/bufferlen = 70/bufferlen = 7
    then @ 7, there are 4 processes/maxpartitions > 7%4 is 3, so it's in the "3rd" (0-index) or "4th" (1-index) subprocess

    REMINDER: THIS ANSWER RETURNS IS BASED ON 0-INDEX!!!
    '''
    testint = args[0]
    bufferlen = args[1]
    maxpartitions = args[2]
    return int(((testint - (testint % bufferlen))/bufferlen)%maxpartitions)

def open_cvpipeline(*args):
    try:
        appliedcv                           = args[0]
        shared_analyzedVAR                  = args[1]
        shared_analyzedKeycountVAR          = args[2]
        # source                              = args[3]
        partitionnumber                     = args[3]
        instance                            = args[4]
        bufferlen                           = args[5]
        maxpartitions                       = args[6]
        fps                                 = args[7]
        shared_rawdict                      = args[8]
        shared_rawKEYSdict                  = args[9]
        FCVAWidget_shared_metadata_dictVAR2 = args[10]

        #didn't know about apipreference: https://stackoverflow.com/questions/73753126/why-does-opencv-read-video-faster-than-ffmpeg
        currentsource = FCVAWidget_shared_metadata_dictVAR2["source"]
        #if source exists (that way you can just start the subprocess w/o requiring a source), if u change source you'll end up triggering the source change code in the while loop so ur good:
        if "source" in FCVAWidget_shared_metadata_dictVAR2.keys():
            sourcecap = cv2.VideoCapture(FCVAWidget_shared_metadata_dictVAR2["source"], apiPreference=cv2.CAP_FFMPEG)
        internal_framecount = 0
        analyzedframecounter = 0
        instance_count = 0
        
        pid = os.getpid()
        FCVAWidget_shared_metadata_dictVAR2["subprocess" + str(pid)] = True

        from collections import deque
        raw_queue = deque(maxlen=bufferlen)
        raw_queueKEYS = deque(maxlen=bufferlen)
        analyzed_queue = deque(maxlen=bufferlen)
        analyzed_queueKEYS = deque(maxlen=bufferlen)

        #init mediapipe here so it spawns the right amt of processes
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        #assume this file structure:
        # this file\examples\creativecommonsmedia\pose_landmarker_full.task is the location
        # https://stackoverflow.com/a/50098973
        from pathlib import Path

        print("file location?", Path(__file__).absolute())
        print("cwd???", os.getcwd())
        if "examples" in os.getcwd().split(os.path.sep):
            # https://stackoverflow.com/a/51276165
            tasklocation = os.path.join(os.sep, os.getcwd().split(os.path.sep)[0] + os.sep, *os.getcwd().split(os.path.sep), "creativecommonsmedia", "pose_landmarker_full.task")
            # tasklocation = os.path.join(os.sep, os.getcwd().split(os.path.sep)[0] + os.sep, *os.getcwd().split(os.path.sep), "creativecommonsmedia", "pose_landmarker_lite.task")
        else:
            tasklocation = 'examples\creativecommonsmedia\pose_landmarker_full.task'
            # tasklocation = 'examples\creativecommonsmedia\pose_landmarker_lite.task'
        fprint("tasklocation?", tasklocation)

        with open(tasklocation, 'rb') as f:
            modelbytes = f.read()
            base_options = python.BaseOptions(model_asset_buffer=modelbytes)
            VisionRunningMode = mp.tasks.vision.RunningMode
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.VIDEO,
                # model_complexity = 0,
                #these were old settings, maybe it's too strict and not giving me poses
                # min_pose_detection_confidence=0.6, min_tracking_confidence=0.6,
                min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
                )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

        while True:
            '''
            PLAN:
            Init shared dicts at the beginning instead of checking every while loop
            
            use 3 subprocesses(A,B,C) to use opencv to get frames from 1 file simultaneously (pray it works and there's no file hold...)
            then for each subprocesses, request 10 frames (0-9 > A, 10-19> B, 20-39>C, etc)
            2 queues, 1 naked frame, 1 analyzed frame that is written to sharedmem for kivy to see
            2 dicts:
            rawqueue
            analyzedqueue

            LOOP:
                3 actions: 
                Write
                    Write to shared dict if init OR frames are old                    
                Analyze
                    Analyze all the time (if analyze queue is empty and there is a framequeue)
                Read
                    request the RIGHT 10 frames (0-10 or 11-20 or 21-30)
                    Load raw frames only if analyze queue is empty (this implicitly checks for time, keeps frames loaded, and stops u from loading too much)
            Why write>analyze>read?
                you want to write out the analyzed frames first
                there is some downtime where kivy reads from a shareddict, in that time I would ideally read/analyze frames (something that doesn't lock the shared dict)
            '''
            #make sure things have started AND this processess is not stopped:

            #if source is different, close cap and reopen with new source: also remember this adds time to this already time critical function...
            if currentsource != FCVAWidget_shared_metadata_dictVAR2["source"]:
                sourcecap.release()
                sourcecap = cv2.VideoCapture(FCVAWidget_shared_metadata_dictVAR2["source"], apiPreference=cv2.CAP_FFMPEG)
                currentsource = FCVAWidget_shared_metadata_dictVAR2["source"]
                # fprint("done? switching?")

            #JUST PAUSE/PLAY RIGHT NOW:
                #press button> togglecv sets the pausetime (when u click pause)
                #UPDATE STARTTIME: 
                #starttime = starttime + time.time - pausetime
                #<seek should happen here but skip for right now>
                #then resume blitting on the proper frame

            #if paused > clear all deques
            #add paused check in if statement below
            #on readframe, add a seek to frame
            #from my quick testing takes ~6ms to get to frame, but doesn't matter since everything should wait until all subprocesses seek to that frame

            if "starttime" in FCVAWidget_shared_metadata_dictVAR2 and ("pausetime" not in FCVAWidget_shared_metadata_dictVAR2) and FCVAWidget_shared_metadata_dictVAR2["subprocess" + str(pid)]:

                initial_time = time.time()
                future_time = FCVAWidget_shared_metadata_dictVAR2["starttime"] + ((1/fps)*internal_framecount)
                current_framenumber = int((time.time() - FCVAWidget_shared_metadata_dictVAR2["starttime"])/(1/fps))
                # fprint("frame advantage START????", os.getpid(), internal_framecount, current_framenumber, future_time-time.time(), time.time())
                
                newwritestart = time.time()
                if len(analyzed_queue) == bufferlen and (max(shared_analyzedKeycountVAR.values()) <= current_framenumber or max(shared_analyzedKeycountVAR.values()) == -1):
                    dictwritetime = time.time()
                    for x in range(bufferlen):
                        shared_analyzedVAR['frame'+str(x)] = analyzed_queue.popleft()
                        shared_analyzedKeycountVAR['key'+str(x)] = analyzed_queueKEYS.popleft()
                    # fprint("updated shareddict", shared_analyzedKeycountVAR.values())
                newwriteend = time.time()
                
                afteranalyzetimestart = time.time()
                if len(raw_queue) > 0 and len(analyzed_queue) == 0:
                    #give the queue to the cv func
                    #cv func returns a queue of frames
                    rtime = time.time()
                    # u can peek at deques: https://stackoverflow.com/questions/48640251/how-to-peek-front-of-deque-without-popping#:~:text=You%20can%20peek%20front%20element,right%20and%20seems%20efficient%20too. , can do it but I thought of a simpler way in the example py file
                    resultqueue = appliedcv(raw_queue, FCVAWidget_shared_metadata_dictVAR2, bufferlen, landmarker, raw_queueKEYS)
                    fprint("resultqueue timing (appliedcv)", os.getpid(), time.time() - rtime, time.time())
                    current_framenumber = int((time.time() - FCVAWidget_shared_metadata_dictVAR2["starttime"])/(1/fps))
                    otherhalf = time.time()

                    #figure out future time
                    future_time = FCVAWidget_shared_metadata_dictVAR2["starttime"] + ((1/fps)*internal_framecount)

                    for x in range(len(resultqueue)):
                        result_compressed = resultqueue.popleft().tobytes()
                        result_compressed = blosc2.compress(result_compressed,filter=blosc2.Filter.SHUFFLE, codec=blosc2.Codec.LZ4)
                        analyzed_queue.append(result_compressed)
                        analyzed_queueKEYS.append(raw_queueKEYS.popleft())
                afteranalyzetime = time.time()

                afterqueuetimestart = time.time()
                # if raw_queue.qsize() == 0:
                # if len(raw_queue) == 0:
                if len(raw_queue) <= int(bufferlen/2):
                    #get the right framecount:
                    framelist = frameblock(partitionnumber,instance_count,bufferlen,maxpartitions)
                    # fprint("says true for some reason?", shared_globalindex_dictVAR["subprocess" + str(pid)])
                    instance_count += 1
                    timeoog = time.time()
                    for x in range(bufferlen*maxpartitions):
                        timegg = time.time()
                        (ret, framedata) = sourcecap.read()  #like .005 speed
                        # fprint("how fast is readin really?", time.time() - timegg) #0.010001897811889648

                        #compare internal framecount to see if it's a frame that this subprocess is supposed to analyze
                        if ret and internal_framecount in framelist:
                            # i might not be picking up a pose because the frame is being read upside down, flip it first before analyzing with mediapipe
                            framedata = cv2.resize(framedata, (1280, 720))
                            # framedata = cv2.resize(framedata, (640, 480))
                            # framedata = cv2.flip(framedata, 0) 
                            # framedata = cv2.cvtColor(framedata, cv2.COLOR_RGB2BGR)
                            raw_queue.append(framedata) #im not giving bytes, yikes? # 0 time
                            raw_queueKEYS.append(framelist[x % bufferlen]) # 0 time
                        internal_framecount += 1
                    # fprint("the for loop structure is slow...", time.time()-timeoog)
    except Exception as e: 
        print("open_appliedcv died!", e)
        import traceback
        print("full exception", "".join(traceback.format_exception(*sys.exc_info())))

class FCVA:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.appliedcv = None

    def run(self):
        try:
            fprint("when compiled, what is __name__?", __name__, "file?", __file__)
            if __name__ == "FastCVApp":
                import multiprocessing as FCVA_mp

                # this is so that only 1 window is run when packaging with pyinstaller
                FCVA_mp.freeze_support()

                # shared_mem_manager = FCVA_mp.Manager()
                
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
                # while True: 
                #     ret, framevar = video.read()
                #     if not ret:
                #         break
                #     self.length += 1
                self.length += 11222333
                video.release()

                kvinit_dict = {}
                #sanity checks
                if not hasattr(self, "fps"):
                    # default to 30fps, else set blit buffer speed to 1/30 sec
                    self.fps = 1 / 30
                if not hasattr(self, "title"):
                    kvinit_dict[
                        "title"
                    ] = "Fast CV App Example v0.1.0 by Pengindoramu"
                else:
                    kvinit_dict["title"] = self.title
                if hasattr(self, "colorfmt"):
                    kvinit_dict["colorfmt"] = self.colorfmt
                if hasattr(self, "kvstring"):
                    kvinit_dict["kvstring"] = self.kvstring
                if self.appliedcv == None:
                    print(
                        "FCVA.appliedcv is currently None. Not starting the CV subprocess."
                    )

                bufferlen = 10
                cvpartitions = 3
                #init shared dicts:

                #nested shared obj works:
                # Everything is shareddict
                # https://bugs.python.org/issue36119
                # nested shared object
                # https://stackoverflow.com/questions/68604215/how-do-you-create-nested-shared-objects-in-multi-processing-in-python

                # the metalist (both shared and not shared) work but are slow: this is most likely because the nested shared dict defeats the purpose of using split shared dicts in that updates happen once instead of multiply at the same time
                # new approach: I'm not smart enough to do this w/o using exec, but generate code on the fly and exec it...
                # reference: https://stackoverflow.com/questions/70862189/how-to-create-variable-names-dynamically-and-assigning-values-in-python
                # reference: https://stackoverflow.com/questions/22558548/eval-syntaxerror-invalid-syntax-in-python

                # shared_pool_meta_list = shared_mem_manager.list()
                # shared_pool_meta_list = [] #IMO this is faster, i think since it doesn't have to propagate changes down the nested dict structure
                # subprocess_list = []
                # initdatalist = FCVA.FCVAWidget_SubprocessInit(
                #     FCVA_mp,
                #     shared_mem_manager,
                #     cvpartitions,
                #     bufferlen,
                #     self.source,
                #     self.fps,
                #     self.appliedcv,
                #     shared_pool_meta_list,
                #     subprocess_list,
                #     )
                # #now set all the stuff that needs to be set from initdatalist:
                # shared_pool_meta_list = initdatalist[0]
                # subprocess_list = initdatalist[1]
                # dicts_per_subprocess =  initdatalist[2]
                
                #you CAN target class methods using multiprocessing process 
                #https://stackoverflow.com/questions/45311398/python-multiprocessing-class-methods
                kivy_subprocess = FCVA_mp.Process(
                    target=self.open_kivy,
                    args=(
                        self.fps,  
                        (1/self.fps), 
                        bufferlen,
                        cvpartitions, 
                        self.length, 
                        kvinit_dict,
                        self.source,
                        self.appliedcv,
                        ))
                kivy_subprocess.start()

                # REMINDER: there is no self because I never instantiate a class with multiprocessing.process
                
                #not necessary
                #new idea: do it vertically: create and init all dicts then run the subprocess
                #when you are done, send all the shared dicts to a list
                #to give the shareddict to kivy subprocess, unpack that list and give the shareddict directly 

                # this try except block holds the main process open so the subprocesses aren't cleared when the main process exits early.
                while True:
                    time.sleep(200)
                    # when the while block is done, close all the subprocesses using .join to gracefully exit. also make sure opencv releases the video.
                    # mediaread_subprocess.join()
                    # for subprocessVAR in subprocess_list:
                    #     subprocessVAR.join()
                    # cv_subprocessA.join()
                    # cv_subprocessB.join()
                    # cv_subprocessC.join()
                    # cv_subprocessD.join()
                    kivy_subprocess.join()
                    fprint("g")
        except Exception as e: 
            print("FCVA run died!", e, flush=True)
            import traceback
            print("full exception", "".join(traceback.format_exception(*sys.exc_info())))

    def FCVAWidget_SubprocessInit(*args): #REMINDER: there is no self because I never instantiate a class with multiprocessing.process
        #more reference.... I can do a class wtf https://stackoverflow.com/questions/17172878/using-pythons-multiprocessing-process-class
        '''
        this is going to spawn subprocesses so make sure the code that calls it has this to stop infinite subprocesses
        if __name__ == "__main__":
            import multiprocessing #edit use multiprocess since it uses dill which apparently is better than pickle as per: https://github.com/ShootingStarDragon/FastCVApp/issues/263
            multiprocessing.freeze_support()
        '''
        FCVA_mpVAR                          = args[0]
        shared_mem_managerVAR               = args[1]
        cvpartitionsVAR                     = args[2]
        bufferlenVAR                        = args[3]
        # sourceVAR                           = args[4] > move this to the shareddict so source can change 
        fpsVAR                              = args[4]
        appliedcvVAR                        = args[5]
        shared_pool_meta_listVAR            = args[6]
        subprocess_listVAR                  = args[7]
        FCVAWidget_shared_metadata_dictVAR  = args[8]
        fprint("check args for FCVAWidget_SubprocessInit", args)

        for x in range(cvpartitionsVAR):
            #init analyzed/keycount dicts
            shared_analyzedA = shared_mem_managerVAR.dict()
            shared_analyzedAKeycount = shared_mem_managerVAR.dict()
            shared_rawA = shared_mem_managerVAR.dict()
            shared_rawAKEYS = shared_mem_managerVAR.dict()
            
            #init dicts
            for y in range(bufferlenVAR):
                shared_analyzedA["frame" + str(y)] = -1
                shared_analyzedAKeycount["key" + str(y)] = -1
                shared_rawA["frame" + str(y)] = -1
                shared_rawAKEYS["key" + str(y)] = -1
            
            #start the subprocesses
            cv_subprocessA = FCVA_mpVAR.Process(
                target=open_cvpipeline,
                args=(
                    appliedcvVAR.__func__, #this is a problem, it doesn't survive multiple dill/pickles...
                    shared_analyzedA,
                    shared_analyzedAKeycount,
                    # sourceVAR,
                    x, #partition #, starts at 0 (now is x in this loop)
                    0, #instance of the block of relevant frames
                    bufferlenVAR, #bufferlen AKA how long the internal queues should be
                    cvpartitionsVAR, #max # of partitions/subprocesses that divide up the video sequence
                    fpsVAR,
                    shared_rawA,
                    shared_rawAKEYS, 
                    FCVAWidget_shared_metadata_dictVAR
                ),
            )
            cv_subprocessA.start()
            #append everything at the end so kivy can start and know all the info
            # thefguy = f'{"shared_analyzed" + str(x) + "OUTERVAR = "} shared_analyzedA'
            # print("thefguy", thefguy)
            # exec(thefguy)
            shared_pool_meta_listVAR.append(shared_analyzedA)
            shared_pool_meta_listVAR.append(shared_analyzedAKeycount)
            shared_pool_meta_listVAR.append(shared_rawA)
            shared_pool_meta_listVAR.append(shared_rawAKEYS)
            dicts_per_subprocessVAR = 4 #remember to update this....
            subprocess_listVAR.append(cv_subprocessA)
        return [shared_pool_meta_listVAR, subprocess_listVAR, dicts_per_subprocessVAR]

    def FCVAWidgetInit(*args, ):#REMINDER: there is no self because I never instantiate a class with multiprocessing.process
        '''
        #1: define class
        #2: set up the kv
        #3: add it to kv string before it's loaded
        
        '''
        from kivy.uix.boxlayout import BoxLayout
        from kivy.clock import Clock
        from kivy.graphics.texture import Texture
        #for drop in (Mac and Windows) #example as per: https://stackoverflow.com/questions/71957402/the-on-drop-file-function-in-kivy-for-python-passes-5-arguments-but-only-3-argu
        from kivy.core.window import Window
        import cv2 #nice, it's ok to load things multiple times python is amazing
        import datetime
        from functools import partial

        class FCVAWidget(BoxLayout):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                #when widget is init start up the subprocesses
                #YOU NEED TO MAKE SURE THE CODE THAT CALLS THIS HAS ALREADY MULTIPROCESSING FREEZE SUPPORT AND IS UNDER SOME GUARD LIKE IF NAME == MAIN
                fprint("what is __name__?", __name__, "this should be bufferlen:", self.bufferlen)
                #in my example I already import multiprocessing. so try if it exist first before I import it twice...
                
                try:
                    FCVA_mp.Manager()
                except Exception as e: 
                    if __name__ == "FastCVApp":
                        import multiprocessing as FCVA_mp
                        FCVA_mp.freeze_support()
                        print("FCVA FCVAWidget __init__ detected no multiprocessing, importing as such", flush=True)
                        # import traceback
                        # print("full exception (YOU CAN IGNORE THIS, just testing if multiprocess/multiprocessing has already been imported)", "".join(traceback.format_exception(*sys.exc_info())))
                
                self.starttime = None
                self.spf = (1/self.fps)

                shared_mem_manager = FCVA_mp.Manager()
                shared_pool_meta_list = [] #IMO this is faster, i think since it doesn't have to propagate changes down the nested dict structure
                subprocess_list = []

                self.FCVAWidget_shared_metadata_dict = shared_mem_manager.dict()
                if hasattr(self, "source"):
                    self.FCVAWidget_shared_metadata_dict["source"] = self.source
                    #sliderdata needs to udpate slider so just schedule for 1st valid frame with clock 0
                    # self.updateSliderData(self.FCVAWidget_shared_metadata_dict)
                    Clock.schedule_once(partial(self.updateSliderData,self.FCVAWidget_shared_metadata_dict), 0)
                    fprint("schedule once???")
                

                initdatalist = FCVA.FCVAWidget_SubprocessInit(
                    FCVA_mp,
                    shared_mem_manager,
                    self.cvpartitions,
                    self.bufferlen,
                    # self.source,
                    self.fps,
                    self.appliedcv,
                    shared_pool_meta_list,
                    subprocess_list,
                    self.FCVAWidget_shared_metadata_dict,
                    )
                #now set all the stuff that needs to be set from initdatalist:
                #put this in the widget for later so I can exit at the end...
                self.shared_pool_meta_list = initdatalist[0]
                self.subprocess_list = initdatalist[1]
                self.dicts_per_subprocess =  initdatalist[2]

                #not sure init has window available so just bind after everything is done using clock schedule once 0
                
                Window.bind(on_drop_file=self._on_file_drop)

            def on_touch_down(self, touch): #overrides touchdown for entire widget
                self.ids['vidsliderID'].on_touch_down(touch) #self is automatically passed i think
                #check if slider is touched as per: https://stackoverflow.com/questions/50590027/how-can-i-detect-when-touch-is-in-the-children-widget-in-kivy and per https://kivy.org/doc/stable/guide/events.html#dispatching-a-property-event
                if self.ids['vidsliderID'].collide_point(*touch.pos):
                    # fprint("touched????", touch)
                    self.toggleCV() #luckily this needs no args

            def on_touch_up(self, touch):
                self.ids['vidsliderID'].on_touch_up(touch)
                # fprint("args???", touch, touch.pos)
                if self.ids['vidsliderID'].collide_point(*touch.pos):
                    fprint("args dont matter, check sliderpos:",self.ids['vidsliderID'].value)
                    self.toggleCV()
                #check if slider has been touched:
                # https://stackoverflow.com/questions/50590027/how-can-i-detect-when-touch-is-in-the-children-widget-in-kivy
                
                # return super().on_touch_move(touch)
            
            def updateSliderData(self, *args):
                '''
                update the slider, right now all it does is update the maxtime by fps * seconds:
                '''
                FCVAWidget_shared_metadata_dictVAR = args[0] 
                sourceguy = FCVAWidget_shared_metadata_dictVAR["source"]
                #https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
                captest = cv2.VideoCapture(sourceguy)
                caplength = int(captest.get(cv2.CAP_PROP_FRAME_COUNT))
                #update slidermax so that u have a 1 to 1 relationship between sliderval and frame:
                self.ids['vidsliderID'].max = caplength
                fprint("what is caplenthg?", caplength)
                capfps = captest.get(cv2.CAP_PROP_FPS)
                captest.release()
                maxseconds = int(caplength/capfps)
                FCVAWidget_shared_metadata_dictVAR["caplength"] = caplength
                FCVAWidget_shared_metadata_dictVAR["capfps"] = capfps
                FCVAWidget_shared_metadata_dictVAR["maxseconds"] = maxseconds
                print( maxseconds )
                # https://stackoverflow.com/questions/775049/how-do-i-convert-seconds-to-hours-minutes-and-seconds
                #so slidermax is a number but you want time like what VLC and Youtube do, so just have a label whose text tracks valuepos and "normalizes" that to the time
                # https://kivy.org/doc/stable/api-kivy.event.html#kivy.event.EventDispatcher.bind
                
                

            def updateSliderMax(self, *args):
                #normalize currentpos against maxframes
                if "caplength" in self.FCVAWidget_shared_metadata_dict:
                    currentpos = args[0]/self.FCVAWidget_shared_metadata_dict["caplength"]
                    # print("what is slider value really??", currentpos)
                    if "maxseconds" in self.FCVAWidget_shared_metadata_dict.keys():
                        # print("what is currentpos??", currentpos)
                        return str(datetime.timedelta(seconds=int(currentpos*self.FCVAWidget_shared_metadata_dict["maxseconds"]))) + "/" + str(datetime.timedelta(seconds=self.FCVAWidget_shared_metadata_dict["maxseconds"]))
                    else:
                        return ""
                else:
                    return ""

            def _on_file_drop(self, window, file_path, x, y):
                print(file_path, str(file_path, encoding='utf-8'))
                self.FCVAWidget_shared_metadata_dict["source"] = str(file_path, encoding='utf-8')
                self.updateSliderData(self.FCVAWidget_shared_metadata_dict)

            def tester(*args):
                fprint("am i accessible in the subprocess after FCVAWidgetInit is called?")
            
            def toggleCV(self, *args):
                # fprint("what are args, do I have widget?, nope, do the search strat", args)
                # fprint("id searching", App.get_running_app().root.get_screen('start_screen_name').ids['FCVAWidget_id'].ids)
                # fprint("id searching", self, self.ids)
                # fprint("id searching", self, self.ids, self.ids['subBoxLayoutID1'], self.ids['subBoxLayoutID1'].ids)
                # widgettext = App.get_running_app().root.get_screen('start_screen_name').ids['FCVAWidget_id'].ids['StartScreenButtonID'].text
                widgettext = self.ids['StartScreenButtonID'].text
                fprint("widgettext is?", widgettext)
                
                #update this play/pause code later
                if "Play" in widgettext:
                    self.ids['StartScreenButtonID'].text = "Pause"
                    if "pausetime" in self.FCVAWidget_shared_metadata_dict.keys():
                        fprint("reset time with pausetime diff:", time.time()- self.FCVAWidget_shared_metadata_dict["pausetime"], "old starttime +3",self.FCVAWidget_shared_metadata_dict["starttime"])

                        self.FCVAWidget_shared_metadata_dict["starttime"] = time.time()- self.FCVAWidget_shared_metadata_dict["pausetime"] + self.FCVAWidget_shared_metadata_dict["starttime"]
                        self.FCVAWidget_shared_metadata_dict.pop("pausetime")
                    else:
                        self.FCVAWidget_shared_metadata_dict["starttime"] = time.time() + 3
                        fprint("set basictime")
                        
                    self.blitschedule = Clock.schedule_interval(self.blit_from_shared_memory, (1/self.fps))
                    
                    #check if you have been paused already:
                    # if "pausedtime" in self.shared_globalindex_dictVAR.keys() and isinstance(self.shared_globalindex_dictVAR["pausedtime"], float):
                    #     #start all subprocesses (hope it's fast enough):
                    #     subprocess_list = [x for x in self.shared_globalindex_dictVAR.keys() if "subprocess" in x]
                    #     for x in subprocess_list:
                    #         self.shared_globalindex_dictVAR[x] = True
                    #     #clear pausedtime and adjust starttime by elapsed time from last pause:
                    #     self.shared_globalindex_dictVAR["starttime"] = self.shared_globalindex_dictVAR["starttime"] + (time.time() - self.shared_globalindex_dictVAR["pausedtime"])
                    #     self.shared_globalindex_dictVAR["pausedtime"] = False
                else:
                    self.ids['StartScreenButtonID'].text = "Play"
                    
                    fprint("set pausetime")
                    self.FCVAWidget_shared_metadata_dict["pausetime"] = time.time()
                    if hasattr(self, "blitschedule"):
                        self.blitschedule.cancel()
                    
                    # self.shared_globalindex_dictVAR["pausedtime"] = time.time()
                    # fprint("#pause all subprocesses (hope it's fast enough):")
                    # subprocess_list = [x for x in self.shared_globalindex_dictVAR.keys() if "subprocess" in x]
                    # for x in subprocess_list:
                    #     self.shared_globalindex_dictVAR[x] = False

                       
                # if "toggleCV" not in self.FCVAWidget_shared_metadata_dict.keys():
                #     self.FCVAWidget_shared_metadata_dict["toggleCV"] = True
                #     if self.starttime == None:
                #         #init starttime:
                #         # self.starttime = time.time() + 1
                #         # self.starttime = time.time() + 2
                #         # self.starttime = time.time() + 3 #wait 3 seconds
                #         self.starttime = time.time() + 8 #need to start 
                #         self.FCVAWidget_shared_metadata_dict["starttime"] = self.starttime
                #         self.index = 0 #this needs to be updated with seek...
                #         self.internal_framecount = 0
                #         self.starttime = None
                #         Clock.schedule_interval(self.blit_from_shared_memory, (1/self.fps))
                #         # start blitting. 1/30 always works because it will always blit the latest image from open_appliedcv subprocess, but kivy itself will be at 30 fps

                # else:
                #     #pop it to remove, that way I can make the time critical stuff faster:
                #     self.FCVAWidget_shared_metadata_dict.pop("toggleCV")

            def populate_texture(self, texture, buffervar):
                texture.blit_buffer(buffervar)
            
            def blit_from_shared_memory(self, *args):
                try:
                    timeog = time.time()
                    # if "toggleCV" in self.FCVAWidget_shared_metadata_dict and self.FCVAWidget_shared_metadata_dict["starttime"] != None:
                    if self.FCVAWidget_shared_metadata_dict["starttime"] != None:
                        self.index = int((time.time() - self.FCVAWidget_shared_metadata_dict["starttime"])/self.spf)
                        # if self.index < 0:
                        #     self.index = 0

                        # fprint("self index>?", self.index)
                        #this is helpful but is very good at locking up the shared dicts...
                        # fprint("is cv subprocess keeping up?", self.index, self.shared_analyzedAKeycountVAR.values(),self.shared_analyzedBKeycountVAR.values(),self.shared_analyzedCKeycountVAR.values(),self.shared_analyzedDKeycountVAR.values())
                        #know the current framenumber
                        #get the right shareddict https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/#
                        # https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
                        # fprint("index in values?A",  self.index, self.shared_analyzedAKeycountVAR.values(), self.index in self.shared_analyzedAKeycountVAR.values())
                        frame = None
                        #hint: u know self.dicts_per_subprocessVAR and self.cvpartitions
                        #this is the nested shared list (containing shared dicts): shared_pool_meta_listVAR
                        #so keycounts are always: 
                        #frameblock(*args):
                        #given partition #, instance, bufferlen, maxpartitions tells u the frames to get:
                        #where partition is x in range(self.cvpartitions), instance is 0, bufferlen is 1, maxpartitions is given by self.cvpartitions

                        # for partitionint in range(self.cvpartitions):
                        #     #note TO FUTURE SELF, THIS LOOKS WRONG, it's it frameblock(partitionint,0,1,self.cvpartitions???) > it's correct, it's a group of 4 and u want the guy in the 1st index (shared_analyzedKeycountIndex)
                        #     shared_analyzedKeycountIndex = frameblock(1,partitionint,1,self.cvpartitions)[0]
                        #     fprint("err here, check numbers","instance",partitionint, "index:", shared_analyzedKeycountIndex,"metalist len", len(self.shared_pool_meta_listVAR))
                        #     fprint("correct index for analyzedkeycount?", self.index, self.shared_pool_meta_listVAR[shared_analyzedKeycountIndex].values())
                        #     shared_analyzedIndex = frameblock(0,partitionint,1,self.cvpartitions)[0]
                        #     if self.index in self.shared_pool_meta_listVAR[shared_analyzedKeycountIndex].values():
                        #         correctkey = list(self.shared_pool_meta_listVAR[shared_analyzedKeycountIndex].keys())[list(self.shared_pool_meta_listVAR[shared_analyzedKeycountIndex].values()).index(self.index)]
                        #         frameref = "frame" + correctkey.replace("key",'')
                        #         frame = self.shared_pool_meta_listVAR[shared_analyzedIndex][frameref]
                        #         break
                        #this doesn't have to be a for loop since u know what index it should be in...
                        #reminder, int to partition is w.r.t. the index and the shared dicts
                        
                        #THIS WORKED
                        shareddict_instance = int_to_partition(self.index,self.bufferlen,self.cvpartitions) 
                        # shared analyzed keycount is w.r.t. getting the right index when the index is self.cvpartitions-many of this sequence: shared_analyzedA, shared_analyzedAKeycount, shared_rawA, shared_rawAKEYS
                        shared_analyzedKeycountIndex = frameblock(1,shareddict_instance,1,self.dicts_per_subprocess)[0] #reminder that frameblock is a continuous BLOCK and shared_pool_meta_listVAR is alternating: 0 1 2 3, 0 1 2 3, etc... which is why bufferlen is 1
                        shared_analyzedIndex = frameblock(0,shareddict_instance,1,self.dicts_per_subprocess)[0]
                        # fprint("valtesting1", self.index, shareddict_instance,shared_analyzedKeycountIndex, len(self.shared_pool_meta_list), shared_analyzedIndex)
                        # fprint("valtesting2", self.index, self.shared_pool_meta_listVAR[shared_analyzedKeycountIndex].values())
                        # fprint("valtesting2", self.index, shared_analyzedKeycountIndex)

                        if self.index in self.shared_pool_meta_list[shared_analyzedKeycountIndex].values():
                            fprint("valtesting3", self.index, list(self.shared_pool_meta_list[shared_analyzedKeycountIndex].values()))
                            correctkey = list(self.shared_pool_meta_list[shared_analyzedKeycountIndex].keys())[list(self.shared_pool_meta_list[shared_analyzedKeycountIndex].values()).index(self.index)]
                            frameref = "frame" + correctkey.replace("key",'')
                            frame = self.shared_pool_meta_list[shared_analyzedIndex][frameref]
                        
                        # https://stackoverflow.com/questions/43748991/how-to-check-if-a-variable-is-either-a-python-list-numpy-array-or-pandas-series
                        
                        if frame != None:
                            frame = blosc2.decompress(frame)
                            # frame = np.frombuffer(frame, np.uint8).copy().reshape(1080, 1920, 3)
                            frame = np.frombuffer(frame, np.uint8).copy().reshape(720, 1280, 3)
                            # frame = np.frombuffer(frame, np.uint8).copy().reshape(480, 640, 3)
                            frame = cv2.flip(frame, 0)
                            buf = frame.tobytes()
                            if isinstance(frame,np.ndarray): #trying bytes
                                #complicated way of safely checking if a value may or may not exist, then get that value:
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

                                ggtime = time.time()
                                self.texture1 = Texture.create(
                                    size=(frame.shape[1], frame.shape[0]), colorfmt=self.colorfmtval
                                )
                                self.texture1.blit_buffer(
                                    buf, colorfmt=self.colorfmtval, bufferfmt="ubyte"
                                )
                                self.ids[
                                    "image_textureID"
                                ].texture = self.texture1
                                # fprint("texture blit entire sequence", time.time()-ggtime) #~8ms... 0.006002187728881836 0.006994962692260742 0.007999658584594727
                                #here update the slider with self.index
                                self.ids['vidsliderID'].value = self.index
                        else:
                            if self.index != 0:
                                # fprint("missed frame#", self.index, self.shared_pool_meta_listVAR[shared_analyzedKeycountIndex].values())
                                fprint("missed frame#", self.index)
                    self.newt = time.time()
                    if hasattr(self, 'newt'):
                        if self.newt - timeog > 0 and (1/(self.newt- timeog)) < 200:
                            # print("blit fps?", 1/(self.newt- timeog))
                            pass
                except Exception as e: 
                    print("blitting died!", e, flush=True)
                    import traceback
                    print("full exception", "".join(traceback.format_exception(*sys.exc_info())))
        
        #change the classdef so that stuff becomes available. This REALLY cannot be called more than once...
        FCVAWidget.cvpartitions = args[0]
        FCVAWidget.bufferlen = args[1]
        FCVAWidget.source = args[2]
        FCVAWidget.fps = args[3]
        FCVAWidget.appliedcv = args[4]

        FCVAWidget_KV = f"""
<FCVAWidget>:
    orientation: 'vertical'
    id: FCVAWidgetID
    Image:
        id: image_textureID
    Slider:
        id: vidsliderID
        min: 0
        max: 100 #will be updated, ideally should be should be 30fps*total_seconds but of course source fps varies BUT imo we'll squish everything to 30fps (or lower, if source is lower)
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
            on_release: FCVAWidgetID.toggleCV()
        Label:
            # text: str(vidsliderID.value) #convert slider label to a time
            text: root.updateSliderMax(vidsliderID.value)
"""
        return FCVAWidget_KV
    
    def open_kivy(*args):
        try:
            # infinite recursion bug when packaging with pyinstaller with no console: https://github.com/kivy/kivy/issues/8074#issuecomment-1364595283
            os.environ["KIVY_NO_CONSOLELOG"] = "1" #logging errs on laptop for some reason
            # if sys.__stdout__ is None or sys.__stderr__ is None:
            #     os.environ["KIVY_NO_CONSOLELOG"] = "1"
            from kivy.app import App
            from kivy.lang import Builder
            from kivy.uix.screenmanager import ScreenManager, Screen
            from kivy.modules import inspector
            from kivy.core.window import Window
            from kivy.uix.button import Button

            class MainApp(App):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    kvinit_dict = self.kvinit_dictVAR
                    kvstring_check = [
                        kvinit_dict[x]
                        for x in kvinit_dict.keys()
                        if x == "kvstring"
                    ]

                    #this loads the class def and sets the kv string as self.FCVAWidget_KV, remember to add self.FCVAWidget_KV to the string
                    # self.FCVAWidgetInit() #this fails because I run this by targeting this function AKA no class exists...
                    self.FCVAWidget_KV = FCVA.FCVAWidgetInit(
                            self.cvpartitions, 
                            self.bufferlen,
                            self.sourceVAR,
                            self.fps,
                            self.appliedcvVAR,
                            )

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
    FCVAWidget:
        id: FCVAWidget_id

FCVA_screen_manager: #remember to return a root widget
"""
                        self.KV_string += self.FCVAWidget_KV     

                def build(self):
                    self.title = self.kvinit_dictVAR["title"]
                    build_app_from_kv = Builder.load_string(self.KV_string)
                    button = Button(text="Test")
                    inspector.create_inspector(Window, button)
                    return build_app_from_kv

                def on_request_close(self, *args):
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

            class FCVA_screen_manager(ScreenManager):
                pass

            class StartScreen(Screen):
                pass

            #since I moved this to a class def all the args got moved by 1 since self is here too
            MainApp.fps                         = args[1]
            MainApp.spf                         = args[2]
            MainApp.bufferlen                   = args[3]
            MainApp.cvpartitions                = args[4]
            MainApp.framelength                 = args[5]
            MainApp.kvinit_dictVAR              = args[6]
            MainApp.sourceVAR                   = args[7]
            MainApp.appliedcvVAR                = args[8]
            
            MainApp().run()
        except Exception as e: 
            print("kivy subprocess died!", e, flush=True)
            import traceback
            print("full exception", "".join(traceback.format_exception(*sys.exc_info())))
