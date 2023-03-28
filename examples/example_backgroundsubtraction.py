import sys
if hasattr(sys, '_MEIPASS'):
    #if file is frozen by pyinstaller add the MEIPASS folder to path:
    sys.path.append(sys._MEIPASS)
    print("meipass should have already been added...")
else:
    #this example is importing from a higher level package if running from cmd: https://stackoverflow.com/a/41575089
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
        print("open_backsub subprocess died! ", e, flush=True)
app.appliedcv = open_backsub

if __name__ == '__main__' :
    app.source = "examples/creativecommonsmedia/Elephants Dream charstart2.webm"
    app.fps = 1/30
    app.title = "Background subtraction example by Pengindoramu"
    app.colorfmt = 'bgr'
    app.kvstring = '''
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
            text: "Start analyzing! This GUI was made with a custom KV string."
            on_release: kivy.app.App.get_running_app().toggleCV()

FCVA_screen_manager: #remember to return a root widget
'''
    app.run()
    