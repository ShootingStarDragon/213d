import os
import time
def fprint(*args):
	print(os.getpid(), time.time(), *args, flush = True)
