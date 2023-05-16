import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('Elephants Dream charstart2.webm')
ret, frame = cap.read()
frame = cv2.resize(frame, (720,1280), interpolation = cv2.INTER_AREA)
# frame = frame.tobytes()
# newframe = np.frombuffer(frame, np.uint8).copy().reshape(1080, 1920, 3)
# newframe.shape > (1080, 1920, 3)
newframe = np.frombuffer(frame, np.uint8).copy().reshape(720, 1280, 3)
# newframe.shape > (1080, 1920, 3)
print("size?", sys.getsizeof(frame), newframe.shape)

