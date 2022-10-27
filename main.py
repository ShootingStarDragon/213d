import kivy
kivy.require('2.1.0') # replace with your current kivy version !

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

builtguy = Builder.load_string('''
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
            ''')

class MyApp(App):

    def __init__(self, *args, **kwargs):
        #remember that KV IS THE ACTUAL FILE AND MUST BE INDENTED PROPERLY WRT TO THE LEFT!
        super().__init__(*args, **kwargs)
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
            '''
        # pass

    def build(self):
        self.title = "Fast CV App v0.1.0 by Pengindoramu"
        build_app_from_kv = Builder.load_string(self.KV_string)
        return build_app_from_kv
        # return builtguy

class FCVA_screen_manager(ScreenManager):
    pass

class StartScreen(Screen):
    pass

if __name__ == '__main__':
    app = MyApp()
    app.run()