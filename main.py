import kivy
kivy.require('2.1.0') # replace with your current kivy version !

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from models.models import texture_format
from typing import Optional

#just in case somebody is using textures before making the app:
'''
If you need to create textures before the application has started, import
    Window first: `from kivy.core.window import Window`
'''
from kivy.core.window import Window

class FastCVApp(App):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #remember that the KV string IS THE ACTUAL FILE AND MUST BE INDENTED PROPERLY TO THE LEFT AS IF IT WERE THE ACTUAL FILE!
        self.KV_string = '''
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
        Label:
            text: "hello world!"

FCVA_screen_manager:
            '''
    def build(self):
        #only build in the main process:
        if __name__ == '__main__':
            self.title = "Fast CV App v0.1.0 by Pengindoramu"
            build_app_from_kv = Builder.load_string(self.KV_string)
            return build_app_from_kv
        else:
            print("Are you sure you're running from __main__? Spawning a subprocess from a subprocess is not ok.")
    #texture documentation: https://github.com/kivy/kivy/blob/master/kivy/graphics/texture.pyx
    #blit buffer example: https://stackoverflow.com/questions/61122285/kivy-camera-application-with-opencv-in-android-shows-black-screen

class FCVA_screen_manager(ScreenManager):
    pass

class StartScreen(Screen):
    pass

# https://stackoverflow.com/questions/5929107/decorators-with-parameters
def CV_function(texture_ref = "None", format: Optional[texture_format] = "RGB"):
    # if texture_ref == "None":
    #     texture_ref = App.get_running_app().root.get_screen('start_screen_name').ids["start_screen_id"].ids["image_textureID"]
    print("texture_ref worksA?", texture_ref, type(texture_ref))
    def wrapper(func, *args, **kwargs):
        def wrapped_f(*args, **kwargs):

            # final_texture = func(*args, **kwargs)
            # #if format is not specified, default to RGB
            print("texture_ref worksB?", texture_ref, type(texture_ref))
            if texture_ref == "None":
                try:
                    #if I use this app MUST be App so that everything works...
                    #next line just kills everything...
                    # texture_ref = App.get_running_app().root.get_screen('start_screen_name').ids["start_screen_id"].ids["image_textureID"]
                    print("x")
                except exception as e:
                    print(e)


            # buf = buf1.tobytes()
            # texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt=texture_format.format) 
            # texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # self.ids["textureB"].texture = texture1

        return wrapped_f
    return wrapper

@CV_function("None")
def testfunc():
    pass

#usage, just decorate their CV_function like so: @CV_function
if __name__ == '__main__':
    app = FastCVApp()
    app.run()
    testfunc()
