import subprocess
import ffmpeg

import time

input_file = "Elephants Dream charstart2FULL.webm"
#dumb idea, open the file as bytes and save to var, then close
 
time1 = time.time()

in_file = open(input_file, "rb") # opening for [r]eading as [b]inary
inputbytes = in_file.read() 
in_file.close()

time2 = time.time()


if time2 - time1 > 0:
    print("spf?", time2 - time1)

args = (ffmpeg
    .input('pipe:')
    # ... extra processing here
    .output('pipe:')
    .get_args()
)
p = subprocess.Popen(['ffmpeg'] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
output_data = p.communicate(input=inputbytes)[0]

# https://stackoverflow.com/questions/23687485/ffmpeg-fails-with-unable-to-find-a-suitable-output-format-for-pipe
# [NULL @ 0000011c10f3a140] Unable to find a suitable output format for 'pipe:'
# pipe:: Invalid argument