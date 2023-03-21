#this example is importing from a higher level package: https://stackoverflow.com/a/41575089
import sys
sys.path.append('../FastCVApp')

import FastCVApp
app = FastCVApp.FCVA()
import cv2

backSub = cv2.createBackgroundSubtractorMOG2()
def open_backsub(*args):
    #reference: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    try:
        image = args[0]
        w_size = (700,500)
        image = cv2.resize(image,w_size)
        image = backSub.apply(image)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        return cv2.flip(image,0)
    except Exception as e:
        print("open_backsub died!", e, flush=True)
app.appliedcv = open_backsub

if __name__ == '__main__' :
    app.source = "creativecommonsmedia/Elephants Dream charstart.webm"
    app.fps = 1/30
    app.title = "Background subtraction example by Pengindoramu"
    app.run()
    