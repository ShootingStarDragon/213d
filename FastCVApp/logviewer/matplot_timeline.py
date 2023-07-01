import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.dates as mdates
from datetime import datetime
import os

#plan read in text file or smth then split > display on matplotlib
print("?????", *__file__.split(os.path.sep)[:-1], )
filelocation = os.path.join(os.sep, __file__.split(os.path.sep)[0] + os.sep, *__file__.split(os.path.sep)[:-1], "logs.txt")
print("filelocation", filelocation)
with open(filelocation, "r", encoding="utf8") as file:
    log_data = file.readlines()

#fix logdata:
log_data_parsed = []
xvarlist = []
yvarlist = []
commentlist = []
for lineVAR in log_data:
    lineVAR_parsed = lineVAR.split(" ")
    if len(lineVAR_parsed) > 1:
        #info is: PID/time in sec/text
        print("index out of range", lineVAR_parsed)
        parsed_time = datetime.fromtimestamp(float(lineVAR_parsed[1])).strftime("%I:%M:%S:%f")
        # print("type wtf", type([lineVAR_parsed[0]]), type([parsed_time]), type(lineVAR_parsed[2:]))
        log_data_parsed.append([lineVAR_parsed[0]] + [parsed_time] + lineVAR_parsed[2:])
        xvarlist.append(parsed_time)
        # yvarlist.append()
        commentlist.append(lineVAR_parsed[2:])

# dates = ['1688087730.719562', '1688087230.719562', '1688087770.719562', '1688087930.719562',]
# dates = [datetime.fromtimestamp(float(VAR)).strftime("%I:%M:%S") for VAR in dates]
# names = ['v2.2.4', 'v3.0.3', 'v3.0.2', 'v3.0.1',]

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
#plot on x axis
ax.plot(xvarlist, np.zeros_like(xvarlist), "-o", color="k", markerfacecolor="w")  # Baseline and markers on it.
# levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(xvarlist)/6)))[:len(xvarlist)]
# levels = np.arrange([-5,5,0.5])
# thearange = np.arange(-5,5,0.5)
# arange = np.array(np.arange(-5,5,1))
# brange = np.array(np.arange(-6,4,1))
# brange = np.array(np.arange(4.5,-4.5,.5))
arange = np.array(np.arange(-5,4.5,0.5))
brange = np.array(np.arange(5,-4.5,-.5))
print(arange, arange.shape)
print(brange, brange.shape)

# https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays-efficiently
def andersonvom(a, b):
    # https://stackoverflow.com/questions/67625331/how-to-fix-must-be-passed-as-a-sequence-warning-for-pil-in-python
    # return np.hstack([var for var in zip(a,b)])
    # https://stackoverflow.com/a/5347082
    return np.vstack((a,b)).reshape((-1,),order='F')
    # return c
thearange = andersonvom(arange, brange)

levels = np.tile(thearange, int(np.ceil(len(xvarlist)/len(thearange))))[:len(xvarlist)]
#plot offset so that matplotlib will resize right
ax.plot(xvarlist, levels, 's',"-o", color="k", markerfacecolor="w", label="testlabel")  # Baseline and markers on it.
#vertical lines
ax.vlines(xvarlist, 0, levels, color="tab:red")
norm = plt.Normalize(vmin=min([lineVAR[1] for lineVAR in log_data]), vmax=max(levels))

#reference: https://github.com/Phlya/adjustText/wiki
from adjustText import adjust_text
annotate_list = []
#annotate writes text and takes (x,y) coords which is why u gotta zip
for nameVAR, dateVAR, levelsVAR in zip(commentlist, xvarlist, levels):
    # annotate_list = ax.annotate(nameVAR, (dateVAR, levelsVAR), textcoords="offset points", horizontalalignment="right",) 
        #verticalalignment="bottom" if l > 0 else "top"
    annotationguy = ax.annotate(nameVAR, (dateVAR, levelsVAR), textcoords="offset points", horizontalalignment="right",) 
    #need this because annotate list complains about non homogeneous size so need to make the list into a single element: " ".join(nameVAR))
    # textelement = ax.text(dateVAR, levelsVAR," ".join(nameVAR))
    # annotationguy.set_visible(False)
    annotate_list.append(annotationguy) 
    
# adjust_text(annotate_list)

# remove y-axis and spines
# ax.yaxis.set_visible(False)
# ax.spines[["left", "top", "right"]].set_visible(False)

# ax.annotate("wtf",(-1, 1), textcoords="offset points", horizontalalignment="right",) 
# from adjustText import adjust_text
# np.random.seed(0)
# x, y = np.random.random((2,30))
# fig, ax = plt.subplots()
# plt.plot(x, y, 'bo')
# texts = [plt.text(x[i], y[i], 'Text%s' %i, ha='center', va='center') for i in range(len(x))]
# adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

#set hover as per: # https://stackoverflow.com/questions/60987714/matplotlib-hover-text
import mplcursors
cursor = mplcursors.cursor(hover=True)
# cursor.connect("add", lambda sel: sel.annotation.set_text(
#     # f"ID:{sel.target.index} '{labels[sel.target.index]}'\nSize:{sizes[sel.target.index]} ({sizes[sel.target.index] * 100.0 / sumsizes:.1f} %)"))
#     f"{dir(sel)[40:]}")) #sel.annotation
# cursor.connect("add", lambda sel: print( dir(sel)))
cursor.connect("add", lambda sel: print( sel.artist.get_label()))

#you can do it right: plan is to set the color<>height correspondence, then when you check selection u can get the right text

#new idea: each pid gets its own curve: https://stackoverflow.com/questions/60209132/display-annotation-text-of-plot-simultaneously-in-matplotlib

plt.show()