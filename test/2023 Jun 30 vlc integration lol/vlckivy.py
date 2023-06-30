#to optimize more, use reload_observer as per: https://stackoverflow.com/questions/51546327/in-kivy-is-there-a-way-to-dynamically-change-the-shape-of-a-texture
#since this is manual stuff (manually setting a buffer in c) ur gonna have to mess with VIDEOHEIGHT or alternative steal it from vlc when it loads a video

# https://stackoverflow.com/questions/37749378/integrate-opencv-webcam-into-a-kivy-user-interface

__author__ = 'bunkus'
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import numpy as np
import vlc
# https://www.geeksforgeeks.org/python-vlc-mediaplayer-taking-screenshot/
# https://github.com/oaubert/python-vlc/issues/66
# import cv2
import os
import ctypes
import sys
from PIL import Image

VIDEOWIDTH = 1920
VIDEOHEIGHT = 1080

# size in bytes when RV32
size = VIDEOWIDTH * VIDEOHEIGHT * 4

# allocate buffer
buf = (ctypes.c_ubyte * size)()
# get pointer to buffer
buf_p = ctypes.cast(buf, ctypes.c_void_p)

# https://github.com/oaubert/python-vlc/issues/17#issuecomment-277476196
CorrectVideoLockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
@CorrectVideoLockCb
def _lockcb(opaque, planes):
    print("lock", file=sys.stderr)
    planes[0] = buf_p

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        # self.capture = cv2.VideoCapture(0)
        # cv2.namedWindow("CV2 Image")
        Clock.schedule_once(self.startup, 0)
        return layout
    
    def startup(self, *args):
        #here we know that everything in kivy is setup after schedule @ time 0, so start the mediaplayer and scan for frames to blit
        self.video_path = '30 fps counter.webm'
        
        #chatgpt trash code
        # # vlc_instance = vlc.Instance('--no-xlib') #chatgpt says --no-xlib, idk why
        # vlc_instance = vlc.Instance()
        # media_player = vlc_instance.media_player_new()

        # media = vlc_instance.media_new(video_path)
        # media_player.set_media(media)

        #load with vlc
        self.media_player = vlc.MediaPlayer()
        media = vlc.Media(os.path.join(os.path.dirname(__file__), self.video_path))
        self.media_player.set_media(media)
        #set the callback: libvlc_video_set_callbacks
        #example: https://github.com/oaubert/python-vlc/issues/17#issuecomment-277476196
        #smth to look at: https://stackoverflow.com/questions/73712284/how-do-i-record-video-on-libvlcpython-binding-python-vlc
        
        
        #get the frame from vlc using libvlc_video_set_callbacks, reference here https://github.com/oaubert/python-vlc/issues/17#issuecomment-277476196
        vlc.libvlc_video_set_callbacks(self.media_player, _lockcb, None, self. update, None)
        #play with vlc
        self.media_player.play()
        #start blitting:
        # Clock.schedule_interval(self.update, 1.0/30.0)
        Clock.schedule_once(self.update, 1.0)
        import time
        # time.sleep(5)


    @vlc.CallbackDecorators.VideoDisplayCb
    def update(self, *args):
        print("args??", args)
        # buf #is the buffer, IS GLOBAL BTW
        # frame = np.frombuffer(buf, np.uint8).copy().reshape(720, 1280, 3)
        img = Image.frombuffer("RGBA", (VIDEOWIDTH, VIDEOHEIGHT), buf, "raw", "BGRA", 0, 1)
        # texture1 = Texture.create(size=(VIDEOWIDTH, VIDEOHEIGHT), colorfmt='bgra') 
        # texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # self.img1.texture = texture1

        #VERY LAZY, DONT MAKE NEW TEXTURES EVERY FRAME, instead use reload_obeserver as per this example:
        # https://stackoverflow.com/questions/51546327/in-kivy-is-there-a-way-to-dynamically-change-the-shape-of-a-texture

        # print("what is this", type(self.framedata))
        # #old example using opencv
        # # display image from cam in opencv window
        # ret, frame = self.capture.read()
        # cv2.imshow("CV2 Image", frame)
        # # convert it to texture
        # buf1 = cv2.flip(frame, 0)
        # buf = buf1.tostring()
        # texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        # #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        # texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # # display image from the texture
        # self.img1.texture = texture1

if __name__ == '__main__':
    CamApp().run()
    # cv2.destroyAllWindows()