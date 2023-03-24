import sys
if hasattr(sys, '_MEIPASS'):
    #if file is frozen by pyinstaller add the MEIPASS folder to path:
    sys.path.append(sys._MEIPASS)
else:
    #this example is importing from a higher level package: https://stackoverflow.com/a/41575089
    sys.path.append('../FastCVApp')

import FastCVApp

app = FastCVApp.FCVA()
import cv2
import numpy as np

def canny_filter(*args):
    #reference: https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
    image = args[0]
    low_threshold = 50
    ratio = 3
    kernel_size = 3

    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = image * (mask[:,:,None].astype(image.dtype))
    return cv2.flip(dst,0)

app.appliedcv = canny_filter

if __name__ == '__main__' :
    app.source = "examples/creativecommonsmedia/Elephants Dream charstart2.webm"
    app.fps = 1/30
    app.title = "Canny edge example by Pengindoramu"
    app.run()
    