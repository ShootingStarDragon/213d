import FastCVApp

app = FastCVApp.FCVA()
import cv2
backSub = cv2.createBackgroundSubtractorMOG2()
def open_backsub(*args):
    #reference: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    try:
        # backSub = cv2.createBackgroundSubtractorKNN()
        # backSub = cv2.createBackgroundSubtractorMOG2()
        image = args[0]
        shared_analysis_dict = args[1]
        shared_metadata_dict = args[2]
        w_size = (700,500)
        image = cv2.resize(image,w_size)
        # # print("image exist?", type(image))
        # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # while True:
        #     if "kivy_run_state" in shared_metadata_dict.keys(): 
        #         if shared_metadata_dict["kivy_run_state"] == False:
        #             break
        #     image = backSub.apply(image)
        #     print("15 sec passed!",type(image), flush= True)
        #     # cv2.imshow('wtfff', image)
        #     shared_analysis_dict[1] = cv2.flip(image,0)
        image = backSub.apply(image)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        #     # shared_analysis_dict[1] = cv2.flip(image,0)
        return cv2.flip(image,0)


#         # image = args[0]
#         # # image = cv2.flip(args[0],1)
#         # shared_analysis_dict = args[1]
#         # shared_metadata_dict = args[2]
#         # # image = backSub.apply(image)
#         # # shared_analysis_dict[1] = cv2.flip(image,0)
#         # # return image
#         # # return image
#         # # shared_analysis_dict[1] = cv2.flip(image,0)
#         # return cv2.flip(image,0)
#         # # print("wtfffff", flush = True)
#         # # print("shared dict appended", type(shared_analysis_dict[1]),flush=True)
    except Exception as e:
        print("open_backsub died!", e, flush=True)
app.appliedcv = open_backsub

# def my_cv_function(inputframe, *args):
#     return inputframe
# app.appliedcv = my_cv_function


if __name__ == '__main__' :
    app.source = "creativecommonsmedia/Elephants Dream charstart.webm"
    app.fps = 1/30
    app.run()
    