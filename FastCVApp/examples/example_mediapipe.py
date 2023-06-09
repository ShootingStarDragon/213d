import sys

if hasattr(sys, "_MEIPASS"):
    # if file is frozen by pyinstaller add the MEIPASS folder to path:
    sys.path.append(sys._MEIPASS)
else:
    # if you're making your own app, you don't need this else block. This is just vanity code so I can run this from main FastCVApp folder or from the examples subfolder.
    # this example is importing from a higher level package if running from cmd: https://stackoverflow.com/a/41575089
    import os

    # add the right path depending on if you're running from examples or from main folder:
    if "examples" in os.getcwd().split(os.path.sep)[-1]:
        sys.path.append(
            ".."
        )  # when running from examples folder, append the upper level
    else:
        # assume they're in main folder trying `python examples/example_backgroundsubtraction.py`
        sys.path.append("../FastCVApp")  # when running from main folder
import FastCVApp

app = FastCVApp.FCVA()
import cv2

# importing here means it's available to the subprocess as well. You can probably cut loading time by only loading mediapipe for the right subprocess.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
def draw_landmarks_on_image(annotated_image, detection_result):
    try:
        pose_landmarks_list = detection_result.pose_landmarks
        
        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        # print("return typoe?", type(annotated_image), len(detection_result.pose_landmarks))
        return annotated_image
    except Exception as e:
        print("open_appliedcv died!", e)
        import traceback
        print("full exception", "".join(traceback.format_exception(*sys.exc_info())))
    
import time
def mediapipe(*args): #basicmp
    try:
        import mediapipe as mp
        from collections import deque
        inputqueue = args[0]
        bufferlenVAR = args[3]
        answerqueue = deque(maxlen=bufferlenVAR)
        landmarkerVAR = args[4]

        while len(inputqueue) > 0:
            
            image = inputqueue.popleft()
            
            ogimage = image.copy()
            image = cv2.resize(image, (256, 144)) #interpolation = cv2.INTER_AREA makes mediapipe detect nothing...
            # image = cv2.resize(image, (640, 360)) #interpolation = cv2.INTER_AREA makes mediapipe detect nothing...
            # image = cv2.resize(image, (640, 480)) #interpolation = cv2.INTER_AREA makes mediapipe detect nothing...
            #so mediapipe is probably legit RGB, but opencv is BGR so convert ONLY for the mediapipe code, but when u draw on the copy of the original things are ok
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Recolor Feed
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            # Make Detections
            # results = detector.detect(image)
            # results = landmarker.detect_for_video(image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            #time has this many digits: 1685543338.9065359, inconsistent digis
            #int(str(time.time())[-10:])
            timestr = str(time.time()).split(".")
            newint = int(timestr[0][-4:]+timestr[1][:3]) #take last 4 of the whole number and first 3 of the decimal, idk if this matters tho
            #time.time should work, i'm feeding them in sequence anyways
            #just making sure they have only the first 3 digits from the decimal and it's an int
            results = landmarkerVAR.detect_for_video(image, newint) 
            # results = landmarkerVAR.detect(image) 
            
            #now draw on original image: 
            fixed_image = draw_landmarks_on_image(ogimage, results)
            answerqueue.append(fixed_image)
        return answerqueue

    except Exception as e:
        print("mediapipe mpvar died!", e, flush=True)
        import traceback
        print("full exception", "".join(traceback.format_exception(*sys.exc_info())))


app.appliedcv = mediapipe

if __name__ == "__main__":
    # / and \ works on windows, only / on mac tho 
    app.source = "examples\creativecommonsmedia\Elephants Dream charstart2FULL.webm"
    app.fps = 1 / 30
    app.title = "Mediapipe example by Pengindoramu"
    app.run()
