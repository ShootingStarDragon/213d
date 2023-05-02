import subprocess
import ffmpeg

input_data = "creativecommonsmedia/Elephants Dream charstart2FULL.webm"
args = (ffmpeg
    .input('pipe:')
    # ... extra processing here
    .output('pipe:')
    .get_args()
)
p = subprocess.Popen(['ffmpeg'] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
output_data = p.communicate(input=input_data)[0]
