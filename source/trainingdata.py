import matplotlib
import matplotlib.figure
import matplotlib.patches
import sqlite3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
#import train_classifier_videofeed
from subprocess import Popen
import sklearn
import datetime
from sklearn import datasets
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
    import ttk

else:
    import tkinter as Tk
    from tkinter import ttk
    from tkinter import filedialog
    from tkinter.ttk import *
import ImageTk
from PIL import Image
from threading import Thread

faces = datasets.fetch_olivetti_faces()

# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================
class Trainer:
    def __init__(self):
        self.results = {}
        self.resultsh = {}
        self.resultss = {}
        self.resultsa = {}
        self.resultssp = {}
        self.resultsn = {}
        self.imgs = faces.images
        self.index = 0

    def reset(self):
        print ("============================================")
        print ("Resetting Dataset & Previous Results.. Done!")
        print ("============================================")
        self.results = {}
        self.resultsh = {}
        self.resultss = {}
        self.resultsa = {}
        self.resultssp = {}
        self.resultsn = {}
        self.imgs = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                # print self.index
                self.index += 1
            return self.index


    def record_result(self, emotion=0):
        if emotion == 1:
            print ("Image", self.index + 1, ":", "Happy")
            self.resultsh[str(self.index)] = True
            self.resultss[str(self.index)] = False
            self.resultsa[str(self.index)] = False
            self.resultssp[str(self.index)] = False
            self.resultsn[str(self.index)] = False
            self.results[str(self.index)] = False
        elif emotion == 2:
            print ("Image", self.index +1, ":", "Sad")
            self.resultsh[str(self.index)] = False
            self.resultss[str(self.index)] = True
            self.resultsa[str(self.index)] = False
            self.resultssp[str(self.index)] = False
            self.resultsn[str(self.index)] = False
            self.results[str(self.index)] = False
        elif emotion == 3:
            print ("Image", self.index +1, ":", "Angry")
            self.resultsh[str(self.index)] = False
            self.resultss[str(self.index)] = False
            self.resultsa[str(self.index)] = True
            self.resultssp[str(self.index)] = False
            self.resultsn[str(self.index)] = False
            self.results[str(self.index)] = False
        elif emotion == 4:
            print ("Image", self.index +1, ":", "Suprised")
            self.resultsh[str(self.index)] = False
            self.resultss[str(self.index)] = False
            self.resultsa[str(self.index)] = False
            self.resultssp[str(self.index)] = True
            self.resultsn[str(self.index)] = False
            self.results[str(self.index)] = False    
        else:
            print ("Image", self.index +1, ":", "Normal")
            self.resultsh[str(self.index)] = False
            self.resultss[str(self.index)] = False
            self.resultsa[str(self.index)] = False
            self.resultssp[str(self.index)] = False
            self.resultsn[str(self.index)] = True
            self.results[str(self.index)] = False


    


# ===================================
# Callback function for the buttons
# ===================================
## smileCallback()              : Gets called when "Happy" Button is pressed
## noSmileCallback()            : Gets called when "Sad" Button is pressed
## updateImageCount()           : Displays the number of images processed
## displayFace()                : Gets called internally by either of the button presses
## displayBarGraph(isBarGraph)  : computes the bar graph after classification is completed 100%
## _begin()                     : Resets the Dataset & Starts from the beginning
## _quit()                      : Quits the Application
## printAndSaveResult()         : Save and print the classification result
## loadResult()                 : Loading the previously stored classification result
## run_once(m)                  : Decorator to allow functions to run only once

def run_once(m):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return m(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

def smileCallback():
    trainer.record_result(emotion=1)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=True, sadCount= False, angryCount= False, suprisedCount = False, normalCount = False)

def noSmileCallback():
    trainer.record_result(emotion=2)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=False, sadCount=True, angryCount= False, suprisedCount = False, normalCount = False)

def angryCallback():
    trainer.record_result(emotion=3)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=False, sadCount=False, angryCount=True, suprisedCount = False, normalCount = False)

def suprisedCallback():
    trainer.record_result(emotion=4)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=False, sadCount=False, angryCount=False, suprisedCount = True, normalCount = False)

def normalCallback():
    trainer.record_result(emotion=5)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=False, sadCount=False, angryCount=False, suprisedCount = False, normalCount = True)


def updateImageCount(happyCount, sadCount, angryCount, suprisedCount, normalCount):
    global HCount, SCount, ACount, SPCount, NCount, imageCountString, countString   # Updating only when called by smileCallback/noSmileCallback
    if happyCount is True and HCount < 400:
        HCount += 1
    if sadCount is True and SCount < 400:
        SCount += 1
    if angryCount is True and ACount < 400:
        ACount += 1
    if suprisedCount is True and SPCount < 400:
        SPCount += 1
    if normalCount is True and NCount < 400:
        NCount += 1
    if HCount == 400 or SCount == 400 or ACount == 400 or SPCount == 400 or NCount == 400:
        HCount = 0
        SCount = 0
        ACount = 0
        NCount = 0
        SPCount = 0

    # --- Updating Labels
    # -- Main Count

    imageCountPercentage = str(float((trainer.index + 1) * 0.25)) \
        if trainer.index+1 < len(faces.images) else "Classification DONE! 100"
    imageCountString = "Image Index: " + str(trainer.index+1) + "/400   " + "[" + imageCountPercentage + " %]"
    labelVar.set(imageCountString)           # Updating the Label (ImageCount)
    # -- Individual Counts
    countString = "(Happy: " + str(HCount) + "   " + "Sad: " + str(SCount) + "   " + "Angry: " + str(ACount) + "   " + "Suprised: " + str(SPCount) + "   " + "Normal: " + str(NCount) + ")\n"
    countVar.set(countString)


@run_once
def displayBarGraph(isBarGraph):
    ax[1].axis(isBarGraph)
    n_groups = 1                    # Data to plot
    Happy, Sad = (sum([trainer.results[x] == True for x in trainer.results]),
               sum([trainer.results[x] == False for x in trainer.results]))
    index = np.arange(n_groups)     # Create Plot
    bar_width = 0.5
    opacity = 0.75
    ax[1].bar(index, Happy, bar_width, alpha=opacity, color='b', label='Happy')
    ax[1].bar(index + bar_width, Sad, bar_width, alpha=opacity, color='g', label='Sad')
    ax[1].set_ylim(0, max(Happy, Sad)+10)
    ax[1].set_xlabel('Expression')
    ax[1].set_ylabel('Number of Images')
    ax[1].set_title('Training Data Classification')
    ax[1].legend()


def printAndSaveResult():
    print (trainer.resultsh)                      # Prints the results
    with open("../results/resultsh.xml", 'w') as output1:
        json.dump(trainer.resultsh, output1)        # Saving The Result

    print (trainer.resultss)                      
    with open("../results/resultss.xml", 'w') as output2:
        json.dump(trainer.resultss, output2) 

    print (trainer.resultsa)                      
    with open("../results/resultsa.xml", 'w') as output3:
        json.dump(trainer.resultsa, output3) 

    print (trainer.resultssp)                      
    with open("../results/resultssp.xml", 'w') as output4:
        json.dump(trainer.resultssp, output4) 

    print (trainer.resultsn)                      
    with open("../results/resultsn.xml", 'w') as output5:
        json.dump(trainer.resultsn, output5) 

def loadResult():
    resultsh = json.load(open("../results/resultsh.xml"))
    trainer.resultsh = resultsh

    resultss = json.load(open("../results/resultss.xml"))
    trainer.resultss = resultss

    resultsa = json.load(open("../results/resultsa.xml"))
    trainer.resultsa = resultsa

    resultssp = json.load(open("../results/resultssp.xml"))
    trainer.resultssp = resultssp

    resultsn = json.load(open("../results/resultsn.xml"))
    trainer.resultsn = resultsn


def displayFace(face):
    printAndSaveResult()

    ax[0].imshow(face, cmap='gray')
    isBarGraph = 'on' if trainer.index+1 == len(faces.images) else 'off'      # Switching Bar Graph ON
    if isBarGraph is 'on':
        displayBarGraph(isBarGraph)
            # f.tight_layout()
    canvas.draw()

def _begin():
    displayFace(trainer.imgs[trainer.index])
    trainer.reset()
    global HCount, SCount, ACount, SPCount, NCount
    HCount = 0
    SCount = 0
    ACount = 0
    SPCount = 0
    NCount = 0
    updateImageCount(happyCount=False, sadCount=False, angryCount=False, suprisedCount = False, normalCount = False)
    


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


if __name__ == "__main__":
    # Embedding things in a tkinter plot & Starting tkinter plot
    matplotlib.use('TkAgg')
    root = Tk.Tk()
    root.wm_title("Emotion Recognition Using Scikit-Learn & OpenCV")

    # =======================================
    # Class Instances & Starting the Plot
    # =======================================
    trainer = Trainer()

    # Creating the figure to be embedded into the tkinter plot
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(faces.images[0], cmap='gray')
    ax[1].axis('off')  # Initially keeping the Bar graph OFF

    # ax tk.DrawingArea
    # Embedding the Matplotlib figure 'f' into Tkinter canvas
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    print ("Keys in the Dataset: ", faces.keys())
    print ("Total Images in Olivetti Dataset:", len(faces.images))

    # Declaring Button & Label Instances
    # =======================================
    smileButton = Tk.Button(master=root, text='Happy', command=smileCallback)
    smileButton.pack(side=Tk.RIGHT)

    noSmileButton = Tk.Button(master=root, text='Sad', command=noSmileCallback)
    noSmileButton.pack(side=Tk.RIGHT)

    angryButton = Tk.Button(master=root, text='Angry', command=angryCallback)
    angryButton.pack(side=Tk.RIGHT)

    angryButton = Tk.Button(master=root, text='Suprised', command=suprisedCallback)
    angryButton.pack(side=Tk.RIGHT)

    angryButton = Tk.Button(master=root, text='Normal', command=normalCallback)
    angryButton.pack(side=Tk.RIGHT)

    labelVar = Tk.StringVar()
    label = Tk.Label(master=root, textvariable=labelVar)
    imageCountString = "Image Index: 0/400   [0 %]"     # Initial print
    labelVar.set(imageCountString)
    label.pack(side=Tk.TOP)

    countVar = Tk.StringVar()
    HCount = 0
    SCount = 0
    ACount = 0
    SPCount = 0
    NCount = 0
    countLabel = Tk.Label(master=root, textvariable=countVar)
    countString = "(Happy: 0   Sad: 0   Angry:0   Suprised:0   Normal:0)\n"     # Initial print
    countVar.set(countString)
    countLabel.pack(side=Tk.TOP)

    resetButton = Tk.Button(master=root, text='Reset', command=_begin)
    resetButton.pack(side=Tk.TOP)

    quitButton = Tk.Button(master=root, text='Quit', command=_quit)
    quitButton.pack(side=Tk.TOP)

    Tk.mainloop()
