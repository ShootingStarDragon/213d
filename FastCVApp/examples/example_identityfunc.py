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
import numpy as np


def sepia_filter2(*args): 
    try:
        # reference: https://medium.com/dataseries/designing-image-filters-using-opencv-like-abode-photoshop-express-part-2-4479f99fb35

        image = args[0]
        # print("who are u?", type(image))
        # image = np.array(image, dtype=np.float64) # converting to float to prevent loss
        # image = cv2.transform(image, np.matrix([[0.272, 0.534, 0.131],
        #                                 [0.349, 0.686, 0.168],
        #                                 [0.393, 0.769, 0.189]]))
        # image[np.where(image > 255)] = 255 # normalizing values greater than 255 to 255
        # image = np.array(image, dtype=np.uint8) # converting back to int
        # # print("what does id func get?", type(image))

        return cv2.flip(image, 0)
        # return image
    except Exception as e:
        print("sepia_filter subprocess died! ", e, flush=True)

'''
https://realpython.com/python-with-statement/#managing-resources-in-python
Call expression to obtain a context manager.
Store the context manager’s .__enter__() and .__exit__() methods for later use.
Call .__enter__() on the context manager and bind its return value to target_var if provided.
Execute the with code block.
Call .__exit__() on the context manager when the with code block finishes.

target_var = expression.__enter__()
do_something(target_var)
expression.__exit__()
'''

# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# importing here means it's available to the subprocess as well. You can probably cut loading time by only loading mediapipe for the right subprocess.
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True).__enter__()
#, static_image_mode=True


# https://stackoverflow.com/questions/51706836/manually-open-context-manager
# Remember that contexts __exit__ method are used for managing errors within the context, so most of them have a signature of __exit__(exception_type, exception_value, traceback), if you dont need to handle it for the tests, just give it some None values:
# mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5).__exit__(None,None,None)

import time

class mediapipeThread:
    def start():
        ..

    def update():
        #the thread looks at an input queue and spits out the output queue

def sepia_filter(*args):
    try:
        raw_queueVAR = args[0]
        shared_globalindex_dictVAR2 = args[1]
        #init mediapipe with/while loop as a thread
            #don't start too many
            # how to store data?, you have to start in subprocess btw, screw it just pass the shared dict info
        #check for thread:
        if "mediapipeThread" + str(os.getpid()) not in shared_globalindex_dictVAR2.keys():
            #start the thread
            mediapipeThread.start()
            shared_globalindex_dictVAR2["mediapipeThread" + str(os.getpid())] = True
        #transfer queue items: raw_queueVAR > new queue
        
        
        
        
        
        # image = args[0]
        time1 = time.time()
        
        frame = args[0]
        frame = image_resize(frame, width = 1280, height = 720)
        
        # print("how long to read frame?", timef2 - timef1)# first frame takes a while and subsequent frames are fast: 0.9233419895172119 -> 0.006009101867675781

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # I have read that writable false/true this makes things faster for mediapipe holistic as per https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

        # Make Detections
        results = holistic.process(image)

        # # Recolor image back to BGR for rendering
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # 2. Right hand
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.right_hand_landmarks,
        #     mp_holistic.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(
        #         color=(80, 22, 10), thickness=2, circle_radius=4
        #     ),
        #     mp_drawing.DrawingSpec(
        #         color=(80, 44, 121), thickness=2, circle_radius=2
        #     ),
        # )

        # # 3. Left Hand
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.left_hand_landmarks,
        #     mp_holistic.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(
        #         color=(121, 22, 76), thickness=2, circle_radius=4
        #     ),
        #     mp_drawing.DrawingSpec(
        #         color=(121, 44, 250), thickness=2, circle_radius=2
        #     ),
        # )

        # # 4. Pose Detections6
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS,
        #     mp_drawing.DrawingSpec(
        #         color=(245, 117, 66), thickness=2, circle_radius=4
        #     ),
        #     mp_drawing.DrawingSpec(
        #         color=(245, 66, 230), thickness=2, circle_radius=2
        #     ),
        # )
        time2 = time.time()
        print("time??? oh shit", time2 - time1, flush= True)
        image = image_resize(image, width = 1920, height = 1080)
        return cv2.flip(image, 0)

        
        
        
        
        # with mp_holistic.Holistic(
        #     min_detection_confidence=0.5, min_tracking_confidence=0.5
        # ) as holistic:

        #     frame = args[0]
        #     # print("how long to read frame?", timef2 - timef1)# first frame takes a while and subsequent frames are fast: 0.9233419895172119 -> 0.006009101867675781

        #     # Recolor Feed
        #     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     image.flags.writeable = False  # I have read that writable false/true this makes things faster for mediapipe holistic

        #     # Make Detections
        #     results = holistic.process(image)

        #     # Recolor image back to BGR for rendering
        #     image.flags.writeable = True
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #     # 2. Right hand
        #     mp_drawing.draw_landmarks(
        #         image,
        #         results.right_hand_landmarks,
        #         mp_holistic.HAND_CONNECTIONS,
        #         mp_drawing.DrawingSpec(
        #             color=(80, 22, 10), thickness=2, circle_radius=4
        #         ),
        #         mp_drawing.DrawingSpec(
        #             color=(80, 44, 121), thickness=2, circle_radius=2
        #         ),
        #     )

        #     # 3. Left Hand
        #     mp_drawing.draw_landmarks(
        #         image,
        #         results.left_hand_landmarks,
        #         mp_holistic.HAND_CONNECTIONS,
        #         mp_drawing.DrawingSpec(
        #             color=(121, 22, 76), thickness=2, circle_radius=4
        #         ),
        #         mp_drawing.DrawingSpec(
        #             color=(121, 44, 250), thickness=2, circle_radius=2
        #         ),
        #     )

        #     # 4. Pose Detections6
        #     mp_drawing.draw_landmarks(
        #         image,
        #         results.pose_landmarks,
        #         mp_holistic.POSE_CONNECTIONS,
        #         mp_drawing.DrawingSpec(
        #             color=(245, 117, 66), thickness=2, circle_radius=4
        #         ),
        #         mp_drawing.DrawingSpec(
        #             color=(245, 66, 230), thickness=2, circle_radius=2
        #         ),
        #     )
        #     time2 = time.time()
        #     print("time??? oh shit", time2 - time1, flush= True)
        #     return cv2.flip(image, 0)


    except Exception as e:
        print("open_mediapipe died!", e, flush=True)
app.appliedcv = sepia_filter

if __name__ == "__main__":
    # / and \ works on windows, only / on mac tho 
    # C:\Personalize\CODING\FastCVApp\fastcvapp\examples\creativecommonsmedia\Elephants Dream charstart2FULL.webm
    # C:\Personalize\CODING\FastCVApp\FastCVApp\examples\creativecommonsmedia\Elephants Dream charstart2.webm
    app.source = "examples\creativecommonsmedia\Elephants Dream charstart2.webm"
    # app.source = "examples\creativecommonsmedia\Elephants Dream 2s.webm"
    # app.source = "examples/creativecommonsmedia/Elephants Dream charstart2FULL.webm"
    # app.source = "examples/creativecommonsmedia/Elephants Dream charstart2.webm"
    # app.source = "examples/creativecommonsmedia/JoJo-s Bizarre Adventure - S05E25 - DUAL 1080p WEB H.264 -NanDesuKa (NF) (1).1080.mp4"
    app.fps = 1 / 30
    app.title = "Sepia filter example by Pengindoramu"
    app.run()
