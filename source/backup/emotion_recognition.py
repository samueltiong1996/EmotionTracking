
#!/usr/bin/env python

"""
======================================================
Emotion recognition using SVMs (Scikit-learn & OpenCV
======================================================

Dependencies: Scikit-Learn,Numpy, Scipy, 
	      Matplotlib, Tkinter

The dataset used in this example is Olivetti Faces:
 http://cs.nyu.edu/~roweis/data/olivettifaces.mat

"""

import matplotlib
import matplotlib.figure
import matplotlib.patches
import sqlite3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
from subprocess import Popen
import sklearn
from sklearn import datasets
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

from tkinter import filedialog

print(__doc__)

with sqlite3.connect('database.db') as db:
    c=db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS graph(happy NUMBER NOT NULL, sad NUMBER NOT NULL);')
db.commit()
db.close()

faces = datasets.fetch_olivetti_faces()

# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================
class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def reset(self):
        print ("============================================")
        print ("Resetting Dataset & Previous Results.. Done!")
        print ("============================================")
        self.results = {}
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

    def record_result(self, smile=True):
        print ("Image", self.index + 1, ":", "Happy" if smile is True else "Sad")
        self.results[str(self.index)] = smile



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

#def smileCallback():
#    trainer.record_result(smile=True)
#    trainer.increment_face()
#    displayFace(trainer.imgs[trainer.index])
#    updateImageCount(happyCount=True, sadCount= False)


#def noSmileCallback():
#    trainer.record_result(smile=False)
#    trainer.increment_face()
#    displayFace(trainer.imgs[trainer.index])
#    updateImageCount(happyCount=False, sadCount=True)


#def updateImageCount(happyCount, sadCount):
#    global HCount, SCount, imageCountString, countString   # Updating only when called by smileCallback/noSmileCallback
#    if happyCount is True and HCount < 400:
#        HCount += 1
#    if sadCount is True and SCount < 400:
#        SCount += 1
#    if HCount == 400 or SCount == 400:
#        HCount = 0
#        SCount = 0
    # --- Updating Labels
    # -- Main Count
#    imageCountPercentage = str(float((trainer.index + 1) * 0.25)) \
#        if trainer.index+1 < len(faces.images) else "Classification DONE! 100"
#    imageCountString = "Image Index: " + str(trainer.index+1) + "/400   " + "[" + imageCountPercentage + " %]"
#    labelVar.set(imageCountString)           # Updating the Label (ImageCount)
#    # -- Individual Counts
#    countString = "(Happy: " + str(HCount) + "   " + "Sad: " + str(SCount) + ")\n"
#    countVar.set(countString)


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


@run_once
def printAndSaveResult():
    print (trainer.results)                      # Prints the results
    with open("../results/results.xml", 'w') as output:
        json.dump(trainer.results, output)        # Saving The Result

@run_once
def loadResult():
    results = json.load(open("../results/results.xml"))
    trainer.results = results


def displayFace(face):
    ax[0].imshow(face, cmap='gray')
    isBarGraph = 'on' if trainer.index+1 == len(faces.images) else 'off'      # Switching Bar Graph ON
    if isBarGraph is 'on':
        displayBarGraph(isBarGraph)
        printAndSaveResult()
    # f.tight_layout()
    canvas.draw()


def _opencv():
    print ("\n\n Please Wait. . . .")
    #execfile("train_classifier_videofeed.py")
    Popen('python train_classifier_videofeed.py', shell='true')
    #$opencvProcess = execfile("train_classifier_videofeed.py")
    #opencvProcess = subprocess.Popen("train_classifier_videofeed.py", close_fds=True, shell=True)
    # os.system('"Train Classifier.exe"')
    # opencvProcess.communicate()

def _linegraph():
    c.execute('SELECT rowid, happy, sad FROM graph')
    ids =[]
    emotion1 = []
    emotion2 = []

    for row in c.fetchall():
        ids.append(row[0])
        emotion1.append(row[1])
        emotion2.append(row[2])

    plt.plot(ids,emotion1,'-')
    plt.plot(ids,emotion2,color='orange')    
    plt.show()

def _begin():
    trainer.reset()
    global HCount, SCount
    HCount = 0
    SCount = 0
    updateImageCount(happyCount=False, sadCount=False)
    displayFace(trainer.imgs[trainer.index])


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


if __name__ == "__main__":
   with sqlite3.connect('database.db') as db:
    c=db.cursor()

    # Embedding things in a tkinter plot & Starting tkinter plot
    matplotlib.use('TkAgg')
    root = Tk.Tk()
    root.wm_title("Emotion Recognition Using Scikit-Learn & OpenCV")

    # =======================================
    # Class Instances & Starting the Plot
    # =======================================
    trainer = Trainer()

    c.execute('SELECT happy,sad FROM graph')
    happy =[]
    sad = []
    total=[]
    color = ['green','blue']

    for row in c.fetchall():
        happy.append(row[0])
        sad.append(row[1])

    total.append(sum(happy))

    total.append(sum(sad))

    fig = matplotlib.figure.Figure(figsize=(5,5))          
    ax = fig.add_subplot(111)
    ax.pie(total)
    ax.legend(['Happy','Sad'])

    # ax tk.DrawingArea
    # Embedding the Matplotlib figure 'f' into Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    canvas.draw()
    opencvButton = Tk.Button(master=root, text='Load the "Trained Classifier" & Test Output', command=_opencv)
    opencvButton.pack(side=Tk.TOP)

    resetButton = Tk.Button(master=root, text='View Line Graph', command=_linegraph)
    resetButton.pack(side=Tk.TOP)

    quitButton = Tk.Button(master=root, text='Quit Application', command=_quit)
    quitButton.pack(side=Tk.TOP)

    authorVar = Tk.StringVar()
    authorLabel = Tk.Label(master=root, textvariable=authorVar)
    authorString = "\n\n Developed By: " \
                   "\n Emotion Tracking Team " \
                   "\n [FYP - Swinburne]"     # Initial print
    authorVar.set(authorString)
    authorLabel.pack(side=Tk.BOTTOM)

   # root.iconbitmap(r'..\icon\happy-sad.ico')
    Tk.mainloop()                               # Starts mainloop required by Tk
