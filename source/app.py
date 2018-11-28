import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
    import ttk

else:
    import tkinter as Tk
    from tkinter import ttk
    from tkinter import filedialog
    from tkinter.ttk import *
    import subprocess
from subprocess import Popen


def cropimage():

    Popen('python emotion_data_prep.py', shell='true')

def imageprocessing():

    Popen('python train_emotion_classifier.py', shell='true')

def main_menu():

    Popen('python emotion_recognition.py', shell='true')


if __name__ == "__main__":

	root = Tk.Tk()
	root.wm_title("Emotion Analytics System")
	root.geometry('400x200')

	labels = Tk.Label(master=root, text='Emotion Analytics System')
	labels.config(font=("Courier", 18))
	labels.pack(side=Tk.TOP)


	cropButton = Tk.Button(master=root, text='Crop Image', command=cropimage)
	cropButton.pack(side=Tk.TOP)

	processButton = Tk.Button(master=root, text='Train Data', command=imageprocessing)
	processButton.pack(side=Tk.TOP)

	mainButton = Tk.Button(master=root, text='Start Menu', command=main_menu)
	mainButton.pack(side=Tk.TOP) 

	Tk.mainloop()
