
#!/usr/bin/env python

"""
======================================================
Emotion recognition using SVMs (Scikit-learn & OpenCV
======================================================

Dependencies: Python, Scikit-Learn, OpenCV,
              Numpy, Scipy, Matplotlib, Tkinter

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

c.execute('CREATE TABLE IF NOT EXISTS graph(times TEXT NOT NULL, happy NUMBER NOT NULL, sad NUMBER NOT NULL);')

c.execute('CREATE TABLE IF NOT EXISTS pie(happynum NUMBER NOT NULL, sadnum NUMBER NOT NULL,total NUMBER NOT NULL);')
db.commit()
db.close()

def _opencv():
    print ("\n\n Please Wait. . . .")
    #execfile("train_classifier_videofeed.py")
    Popen('python train_classifier_videofeed.py', shell='true')
    #$opencvProcess = execfile("train_classifier_videofeed.py")
    #opencvProcess = subprocess.Popen("train_classifier_videofeed.py", close_fds=True, shell=True)
    # os.system('"Train Classifier.exe"')
    # opencvProcess.communicate()

def _trainingdata():
    Popen('python trainingdata.py', shell='true')

def _linegraph():
    c.execute('SELECT rowid, happy, sad FROM graph')
    ids =[]
    emotion1 = []
    emotion2 = []

    for row in c.fetchall():
        ids.append(row[0])
        emotion1.append(row[1])
        emotion2.append(row[2])

    plt.plot(ids,emotion1,color='green')
    #plt.plot(ids,emotion2,color='orange')   
    plt.xlabel('Time')
    plt.ylabel('Percentage (%)') 
    plt.show()

def _piechart():
    c.execute('SELECT happynum,sadnum,total FROM pie')
    happy =[]
    sad = []
    datatotal=[]
    totals=[]
    color = ['green','blue']
    label = ['happy', 'not happy']

    for row in c.fetchall():
        happy.append(row[0])
        sad.append(row[1])
        datatotal.append(row[2])

    totals.append((sum(happy))/(sum(datatotal)))

    totals.append((sum(sad))/(sum(datatotal)))

    plt.pie(totals,labels=label,colors=color,autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Overall Customer Emotion', bbox={'facecolor':'0.8','pad':5})
    fig = plt.gcf()         
    fig.add_subplot(111)
    fig.set_size_inches(3,3)

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


    labels = Tk.Label(master=root, text='Emotion Analytics System')
    labels.config(font=("Courier", 18))
    labels.pack(padx=10,pady=50,side=Tk.TOP)

    traindataButton = Tk.Button(master=root, text='Train data', command=_trainingdata)
    traindataButton.pack(side=Tk.TOP)

    opencvButton = Tk.Button(master=root, text='Livefeed Cam', command=_opencv)
    opencvButton.pack(side=Tk.TOP)

    lineButton = Tk.Button(master=root, text='View Line Graph', command=_linegraph)
    lineButton.pack(side=Tk.TOP)

    pieButton = Tk.Button(master=root, text='Pie Chart', command=_piechart)
    pieButton.pack(side=Tk.TOP)

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
