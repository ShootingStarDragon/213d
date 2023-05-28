# https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import time
# For webcam input:
cap = cv2.VideoCapture("Elephants Dream charstart2.webm")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    time1 = time.time()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    time2 = time.time()
    print("is mp dropping frames? > by itself it's reporting ~0.05 sec per frame, 44 percent cpu usage",time2-time1 ,flush=True)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()





# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:



# while cap.isOpened():
#     pose = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5).__enter__()
#     #, static_image_mode=True
#     time1 = time.time()
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     time2 = time.time()
#     print("is mp dropping frames? > not closing the context really hurts, goes from 0.05 to 0.1",time2-time1 ,flush=True)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
#     pose.__exit__(None,None,None) #-> manually exiting sets it to 0.3seconds, how to do this right?
# cap.release()