
# https://kkroening.github.io/ffmpeg-python/

import ffmpeg
import cv2
import numpy as np

in_filename = "Elephants Dream charstart2.webm" 

in_file = open(in_filename, "r") # opening for [r]eading as [b]inary
inputbytes = in_file.read() 
in_file.seek(0) #wtf, remember to seek to beginning??
in_file.close()

process1 = (
    ffmpeg
    .input(in_filename)
    # .input('pipe:')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

process1a = (
    ffmpeg
    .input(in_filename)
    # .input('pipe:')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

process1b = (
    ffmpeg
    .input(in_filename)
    # .input('pipe:')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

width, height = 1920, 1080

while True:
    # process1.communicate(input=inputbytes)[0]
    # The whole purpose of the communicate method is to wait for the process to finish and return all the output
    # https://stackoverflow.com/questions/2133345/python-subprocess-communicate-block

    # https://stackoverflow.com/questions/8475290/how-do-i-write-to-a-python-subprocess-stdin

    intoffmpeg = process1.stdin.write(inputbytes[:width * height * 3])

    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )
    # out_frame = in_frame * 0.3
    # https://www.scivision.dev/numpy-image-bgr-to-rgb/
    in_frame = in_frame[...,[2, 1, 0]].copy() # this is correct
    cv2.imshow('img', in_frame)  # Show the image for testing
    # cv2.waitKey(1000)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

process1.wait()
