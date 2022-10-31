import cv2.data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

import dill
print("dill pickles!", dill.pickles(face_cascade)) 
