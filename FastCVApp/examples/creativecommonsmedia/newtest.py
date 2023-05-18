import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('Elephants Dream charstart2.webm')
ret, frame = cap.read()
# frame = cv2.resize(frame, (720,1280), interpolation = cv2.INTER_AREA)
# frame = frame.tobytes()
# newframe = np.frombuffer(frame, np.uint8).copy().reshape(1080, 1920, 3)
# newframe.shape > (1080, 1920, 3)
# newframe = np.frombuffer(frame, np.uint8).copy().reshape(720, 1280, 3)
# newframe.shape > (1080, 1920, 3)
# print("size?", sys.getsizeof(frame), newframe.shape)
print("size?", sys.getsizeof(frame))


# use the io library
# https://stackoverflow.com/questions/44672524/how-to-create-in-memory-file-object

#trying out np.save:
# https://stackoverflow.com/questions/25837641/save-retrieve-numpy-array-from-string
# https://stackoverflow.com/a/25837662
# savez saves the array in npz format (uncompressed)
# savez_compressed saves the array in compressed npz format
# savetxt formats the array in a humanly readable format

import io, time
filelike = io.BytesIO() #this is still bytes I think?
time1 = time.time()
np.savez_compressed(filelike, frame=frame) #numpy takes too long.. filelike size? 3207416 time? 0.13618159294128418
time2 = time.time()
filelike.seek(0)
print("filelike size?", sys.getsizeof(filelike), "time?", time2-time1 )

# loaded = np.load(filelike.getvalue())
timea = time.time()
loaded = np.load(filelike,allow_pickle=True)
timeb = time.time()
print("filelike decompress?", sys.getsizeof(loaded), timeb-timea)

# while True:
#     cv2.imshow('img', loaded['frame'])  # Show the image for testing
#     # cv2.waitKey(1000)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
filelike.close()

#try blosc:
# as per: https://www.blosc.org/python-blosc/tutorial.html
# import numpy as np
import blosc2
a = np.linspace(0, 100, 10000000) 
a.astype(int) #https://www.statology.org/numpy-array-to-int/
bytes_array = a.tobytes()  # get an array of bytes (tostring deprecated, use tobytes)

# import timeit
time1 = time.time()
blosc2.compress(bytes_array, typesize=8)
time2 = time.time()
print("blosc2 time on random array?", time2 - time1)
# print(timeit.timeit('blosc2.compress(bytes_array, typesize=8)'))
# try these: 

ret, frame2 = cap.read()
timeA = time.time()
framebytes = frame2.tobytes()
timeB = time.time()

time1 = time.time()
blosc2.compress(frame2, typesize=8)
time2 = time.time()
print("blosc2 time on my framedata?", time2 - time1, "time to tobytes", timeB - timeA, "framesize", sys.getsizeof(framebytes))

time1 = time.time()
# c = blosc2.pack(frame2.__array_interface__['data'][0], frame2.size, frame2.dtype.itemsize, 9, True)
c = blosc2.pack(frame2)
time2 = time.time()
print("blosc2 compress pointer time", time2 - time1, "blosc2 array size?", sys.getsizeof(c))
#how to decompress:
# blosc.unpack(c, a2.__array_interface__['data'][0])
# https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md
# The functions compress_ptr and decompress_ptr are replaced by pack and unpack since Pickle protocol 5 comes with out-of-band data.

time1 = time.time()
undoneframe = blosc2.unpack(c)
time2 = time.time()

print("time to undo blosc2?", time2-time1, type(c)) #0.004288196563720703
print("pyversion", sys.version)
while True:
    cv2.imshow('img', undoneframe)  # Show the image for testing
    # cv2.waitKey(1000)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
