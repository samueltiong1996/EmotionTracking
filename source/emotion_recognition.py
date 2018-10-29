
#!/usr/bin/env python

#"""
#======================================================
#Emotion recognition using SVMs (Scikit-learn & OpenCV
#======================================================

#Dependencies: Python 2.7, Scikit-Learn, OpenCV 3.0.0,
#              Numpy, Scipy, Matplotlib, Tkinter
#Instructions: Please checkout Readme.txt & Instructions.txt

#The dataset used in this example is Olivetti Faces:
# http://cs.nyu.edu/~roweis/data/olivettifaces.mat

#"""

import matplotlib
import matplotlib.figure
import matplotlib.patches
import sqlite3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
import train_classifier_videofeed
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
from multiprocessing import Process

print(__doc__)

with sqlite3.connect('database.db') as db:
    c=db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS TEMP(PRODUCT_CODE TEXT NOT NULL,times TEXT NOT NULL, happy NUMBER NOT NULL, sad NUMBER NOT NULL, angry NUMBER NOT NULL, suprised NUMBER NOT NULL, normal NUMBER NOT NULL);')
c.execute('CREATE TABLE IF NOT EXISTS EMOTION(PRODUCT_CODE TEXT NOT NULL,DATES TEXT NOT NULL, daily_happy NUMBER NOT NULL, daily_sad NUMBER NOT NULL, daily_angry NUMBER NOT NULL, daily_suprised NUMBER NOT NULL, daily_normal NUMBER NOT NULL);')
c.execute('CREATE TABLE IF NOT EXISTS PRODUCT(PRODUCT_CODE TEXT NOT NULL,product_brand TEXT NOT NULL,product_category TEXT NOT NULL, product_name TEXT NOT NULL);')

db.commit()


faces = datasets.fetch_olivetti_faces()

# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================
#class Trainer:

    #train_classifier_videofeed.main()

    #def __init__(self):
    #    self.results = {}
    #    self.imgs = faces.images
    #    self.index = 0

    #def reset(self):
    #    print ("============================================")
    #    print ("Resetting Dataset & Previous Results.. Done!")
    #    print ("============================================")
    #    self.results = {}
    #    self.imgs = faces.images
    #    self.index = 0

    #def increment_face(self):
    #    if self.index + 1 >= len(self.imgs):
    #        return self.index
    #    else:
    #        while str(self.index) in self.results:
                # print self.index
    #            self.index += 1
    #        return self.index

    #def record_result(self, smile=True):
    #    print ("Image", self.index + 1, ":", "Happy" if smile is True else "Sad")
    #    self.results[str(self.index)] = smile



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

#def run_once(m):
#    def wrapper(*args, **kwargs):
#        if not wrapper.has_run:
#            wrapper.has_run = True
#            return m(*args, **kwargs)
#    wrapper.has_run = False
#    return wrapper

#def smileCallback():
#    trainer.record_result(smile=True)
#    trainer.increment_face()
#    displayFace(trainer.imgs[trainer.index])
#    updateImageCount(happyCount=True, sadCount= False)


#def noSmileCallback():
 #   trainer.record_result(smile=False)
 #   trainer.increment_face()
 #   displayFace(trainer.imgs[trainer.index])
 #   updateImageCount(happyCount=False, sadCount=True)


#def updateImageCount(happyCount, sadCount):
#    global HCount, SCount, imageCountString, countString   # Updating only when called by smileCallback/noSmileCallback
#    if happyCount is True and HCount < 400:
#        HCount += 1
#    if sadCount is True and SCount < 400:
#        SCount += 1
#    if HCount == 400 or SCount == 400:
#        HCount = 0
 #       SCount = 0
    # --- Updating Labels
    # -- Main Count
#    imageCountPercentage = str(float((trainer.index + 1) * 0.25)) \
#        if trainer.index+1 < len(faces.images) else "Classification DONE! 100"
#    imageCountString = "Image Index: " + str(trainer.index+1) + "/400   " + "[" + imageCountPercentage + " %]"
#    labelVar.set(imageCountString)           # Updating the Label (ImageCount)
#    # -- Individual Counts
#    countString = "(Happy: " + str(HCount) + "   " + "Sad: " + str(SCount) + ")\n"
#    countVar.set(countString)


#@run_once
#def displayBarGraph(isBarGraph):
#    ax[1].axis(isBarGraph)
#    n_groups = 1                    # Data to plot
#    Happy, Sad = (sum([trainer.results[x] == True for x in trainer.results]),
#               sum([trainer.results[x] == False for x in trainer.results]))
#    index = np.arange(n_groups)     # Create Plot
#    bar_width = 0.5
#    opacity = 0.75
#    ax[1].bar(index, Happy, bar_width, alpha=opacity, color='b', label='Happy')
#    ax[1].bar(index + bar_width, Sad, bar_width, alpha=opacity, color='g', label='Sad')
#    ax[1].set_ylim(0, max(Happy, Sad)+10)
#    ax[1].set_xlabel('Expression')
#    ax[1].set_ylabel('Number of Images')
#    ax[1].set_title('Training Data Classification')
#    ax[1].legend()


#@run_once
#def printAndSaveResult():
#    print (trainer.results)                      # Prints the results
#    with open("../results/results.xml", 'w') as output:
#        json.dump(trainer.results, output)        # Saving The Result

#@run_once
#def loadResult():
#    results = json.load(open("../results/results.xml"))
#    trainer.results = results


#def displayFace(face):
#    ax[0].imshow(face, cmap='gray')
#    isBarGraph = 'on' if trainer.index+1 == len(faces.images) else 'off'      # Switching Bar Graph ON
#    if isBarGraph is 'on':
#        displayBarGraph(isBarGraph)
#        printAndSaveResult()
    # f.tight_layout()
#    canvas.draw()


def _opencv():

    #print ("\n\n Please Wait. . . .")
    #execfile("train_classifier_videofeed.py")
    Popen('python trainingdata.py', shell='true')
    #$opencvProcess = execfile("train_classifier_videofeed.py")
    #opencvProcess = subprocess.Popen("train_classifier_videofeed.py", close_fds=True, shell=True)
    # os.system('"Train Classifier.exe"')
    # opencvProcess.communicate()

def _linegraph(combo2):

    a.clear()
    linechoice = 'SELECT * FROM TEMP WHERE PRODUCT_CODE = ?'
    c.execute(linechoice,[combo2.get()])
    ids =[]
    timest = []
    ehappy = []
    esad = []
    eangry = []
    esuprised = []
    enormal = []

    for row in c.fetchall():
        ids.append(row[0])
        timest.append(row[1])
        ehappy.append(row[2])
        esad.append(row[3])
        eangry.append(row[4])
        esuprised.append(row[5])
        enormal.append(row[6])

    a.set_xlabel("Time")
    a.set_ylabel("Emotions") 
    
    d1 = a.plot(timest,ehappy,color="green",label="Happy")
    d2 = a.plot(timest,esad,color="blue",label="Sad")
    d3 = a.plot(timest,eangry,color="red",label="Angry")
    d4 = a.plot(timest,esuprised,color="purple",label="Suprised")
    d5 = a.plot(timest,enormal,color="grey",label="Normal")

    data = [d1,d2,d3,d4,d5]
    a.legend()
    canvas.draw()

def _barchart(combo1,itemlist):

    ids =[]
    timesd = []
    happy = []
    sad = []
    angry = []
    suprised = []
    normal = []

    a2.clear()
    global countx 
    countx = 0
    searchdb = 'SELECT * FROM PRODUCT WHERE product_brand = ?'
    c.execute(searchdb,[combo1.get()])

    for row in c.fetchall():
        dailyrecord2 = 'SELECT * FROM EMOTION WHERE PRODUCT_CODE = ?'
        c.execute(dailyrecord2,[row[0]])

        for row in c.fetchall():
            ids.append(row[0])
            timesd.append(row[1])
            happy.append(row[2])
            sad.append(row[3])
            angry.append(row[4])
            suprised.append(row[5])
            normal.append(row[6])

        countx = countx + 1
      
    countx = np.arange(countx)

    a2.bar(countx,happy,0.10, alpha=0.8,color='green',label='Happy')
    a2.bar(countx+0.10,sad,0.10, alpha=0.8, color='blue',label='Sad')
    a2.bar(countx+0.20,angry,0.10, alpha=0.8, color='red',label='Angry')
    a2.bar(countx+0.30,suprised,0.10, alpha=0.8, color='purple',label='Surprised')
    a2.bar(countx+0.40,normal,0.10, alpha=0.8, color='grey',label='Normal')

    a2.set_xlabel('Product')
    a2.set_ylabel('Total Emotion')
    a2.set_xticks(countx)
    a2.set_xticklabels(ids)
    a2.legend()
    canvas2.draw()

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

def _product(page):
    addproduct = Tk.Toplevel(root)
    addproduct.wm_title("Add Product")

    pcode = Tk.Entry(addproduct)
    pbrand = Tk.Entry(addproduct)
    pcategory = Tk.Entry(addproduct)
    pname = Tk.Entry(addproduct)

    Tk.Label(addproduct, text="Product Code").grid(row=0)
    Tk.Label(addproduct, text="Product Brand").grid(row=1)
    Tk.Label(addproduct, text="Product Category").grid(row=2)
    Tk.Label(addproduct, text="Product Name").grid(row=3)

    pcode.grid(row=0, column=1)
    pbrand.grid(row=1, column=1)
    pcategory.grid(row=2, column=1)
    pname.grid(row=3, column=1)

    addbutton = Tk.Button(addproduct, text='Add Product', command=lambda:ptodb(page,addproduct,pcode,pbrand,pcategory,pname)).grid(row=4, column=0, sticky='W', pady=4)
    Tk.Button(addproduct, text='Exit', command=addproduct.destroy).grid(row=4, column=1, sticky='W', pady=4)    
    addproduct.transient(root)
    addproduct.grab_set()
    root.wait_window(addproduct)

def ptodb(page,f,p1,p2,p3,p4):
    valid = False
    c.execute('SELECT PRODUCT_CODE FROM PRODUCT')
    for row in c.fetchall():
        if p1.get()==row[0]:
            
            valid = True
            break

        else:   
            valid = False

    if valid==False:        
        saveproduct = 'INSERT INTO PRODUCT(PRODUCT_CODE,product_brand,product_category,product_name) VALUES (?,?,?,?)'
        c.execute(saveproduct,[p1.get(),p2.get(),p3.get(),p4.get()])
    
        db.commit()

        for i in page.get_children():
            page.delete(i)


        c.execute('SELECT * FROM PRODUCT')
        productdata = c.fetchall()



        index = iid = 0
        for row in productdata:
            page.insert("", index, iid, values = row)
            index = iid = index + 1

        f.destroy()
    
    else:
        Tk.Label(f, text="Error:Duplicate Product Code").grid(row=6)

def deletep(tree):
    for selected_item in tree.selection():
        c.execute('DELETE from PRODUCT WHERE PRODUCT_CODE=?',(tree.set(selected_item,'#1'),))
        db.commit()
        tree.delete(selected_item)


def editp(tree):

    for selected_item in tree.selection():
        c.execute('SELECT * from PRODUCT WHERE PRODUCT_CODE=?',(tree.set(selected_item,'#1'),))
        
    if selected_item is not None:

        for row in c.fetchall():
            p1 = row[0]
            p2 = row[1]
            p3 = row[2]
            p4 = row[3]


        editproduct = Tk.Toplevel(root)
        editproduct.wm_title("Edit Product")

        pcode = Tk.Entry(editproduct)
        pbrand = Tk.Entry(editproduct)
        pcategory = Tk.Entry(editproduct)
        pname = Tk.Entry(editproduct)

        Tk.Label(editproduct, text="Product Code").grid(row=0)
        Tk.Label(editproduct, text="Product Brand").grid(row=1)
        Tk.Label(editproduct, text="Product Category").grid(row=2)
        Tk.Label(editproduct, text="Product Name").grid(row=3)

        pcode.grid(row=0, column=1)
        pbrand.grid(row=1, column=1)
        pcategory.grid(row=2, column=1)
        pname.grid(row=3, column=1)

        pcode.insert(0,p1)
        pcode.configure(state='readonly')
        pbrand.insert(0,p2)
        pcategory.insert(0,p3)
        pname.insert(0,p4)

        editbutton = Tk.Button(editproduct, text='Edit Product', command=lambda:updatep(tree,editproduct,pcode,pbrand,pcategory,pname)).grid(row=4, column=0, sticky='W', pady=4)
        Tk.Button(editproduct, text='Exit', command=editproduct.destroy).grid(row=4, column=1, sticky='W', pady=4)    
        editproduct.transient(root)
        editproduct.grab_set()
        root.wait_window(editproduct)


def updatep(page,f,p1,p2,p3,p4):
    updateproduct = 'UPDATE PRODUCT SET product_brand = ?, product_category = ?, product_name = ? WHERE PRODUCT_CODE = ?'
    c.execute(updateproduct,[p2.get(),p3.get(),p4.get(),p1.get()])
    
    db.commit()

    for i in page.get_children():
        page.delete(i)

    c.execute('SELECT * FROM PRODUCT')
    productdata = c.fetchall()

    index = iid = 0
    for row in productdata:
        page.insert("", index, iid, values = row)
        index = iid = index + 1

    f.destroy()

def _savedaily():
    global thappy
    global tsad
    global tangry
    global tsuprised
    global tnormal
    global TF
    TF = False
    thappy = 0
    tsad = 0
    tangry = 0
    tsuprised = 0
    tnormal = 0


    datetoday = datetime.datetime.now().date()
    checkdate = 'SELECT * FROM EMOTION'
    c.execute(checkdate)    

    #for row in c.fetchall():
     #   if datetoday == row[1]:
     #       TF = True
     #       break
     #   else:
     #       TF = False


    #if TF == False:

    getcode = 'SELECT DISTINCT PRODUCT_CODE FROM TEMP'
    c.execute(getcode)
    for row1 in c.fetchall():
        thappy = 0
        tsad = 0
        tangry = 0
        tsuprised = 0
        tnormal = 0
        saved = 'SELECT * FROM TEMP'
        c.execute(saved)
        for row in c.fetchall():
            if row[0] == row1[0]:
                thappy = thappy + row[2]
                tsad = tsad + row[3]
                tangry = tangry + row[4]
                tsuprised = tsuprised + row[5]
                tnormal = tnormal + row[6]

        savedb = 'INSERT INTO EMOTION(PRODUCT_CODE, DATES, daily_happy, daily_sad, daily_angry, daily_suprised, daily_normal) VALUES (?,?,?,?,?,?,?)'
        c.execute(savedb,[row1[0],datetoday,thappy,tsad,tangry,tsuprised,tnormal])
        db.commit()


    c.execute('SELECT * FROM EMOTION')
    productdata1 = c.fetchall()
    index1 = iid2 = 0
    for row in productdata1:
        newtree1.insert("", index1, iid2, values = row)
        index1 = iid2 = index1 + 1

    #delrecord = 'DELETE FROM TEMP'
    #c.execute(delrecord)
    #db.commit()

def _new(choice):
    startto = Process(target = lambda:train_classifier_videofeed.main(choice.get()))
    startto.start()


def refreshproductname():
    c.execute('SELECT * FROM EMOTION')
    productdata1 = c.fetchall()
    index1 = iid2 = 0
    for row in productdata1:
        newtree1.insert("", index1, iid2, values = row)
        index1 = iid2 = index1 + 1




if __name__ == "__main__":
   with sqlite3.connect('database.db') as db:
    c=db.cursor()

    product1=[]
    product2=[]
    product3=[]
    product4=[]

    # Embedding things in a tkinter plot & Starting tkinter plot
    
    matplotlib.use('TkAgg')
    root = Tk.Tk()
    root.wm_title("Emotion Analytics System")
    root.geometry('900x700')

    rows = 0
    while rows < 50:
        root.rowconfigure(rows, weight=1)
        root.columnconfigure(rows, weight=1)
        rows +=1

    nb = ttk.Notebook(root)
    nb.grid(row=1,column=0, columnspan=50, rowspan=49,sticky='NEWS')

    

    page1 = ttk.Frame(nb)
    nb.add(page1,text='Camera')

    page2 = ttk.Frame(nb)
    nb.add(page2,text='Bar Graph')

    page3 = ttk.Frame(nb)
    nb.add(page3,text='Line')
   
    page4 = ttk.Frame(nb)
    nb.add(page4,text='Product')

    # =======================================
    # Class Instances & Starting the Plot
    # =======================================
    #trainer = Trainer()

    labels = Tk.Label(master=root, text='')
    labels.config(font=("Courier", 18))

    #Page1
    newtree1 = ttk.Treeview(page1)
    newtree1.pack(padx=10,pady=10)

    newtreebar1 = ttk.Scrollbar(page1, orient="horizontal", command = newtree1.xview)
    newtreebar1.pack(side='bottom', fill='x')

    newtree1.configure(xscrollcommand=newtreebar1.set)

    newtree1["columns"] = ["Product Code","Date", "Happy", "Sad", "Angry","Suprised","Normal"]    
    newtree1["show"] = "headings"
    newtree1.heading("Product Code", text ="Product Code")
    newtree1.heading("Date", text ="Date")
    newtree1.heading("Happy", text ="Happy")
    newtree1.heading("Sad", text ="Sad")
    newtree1.heading("Angry", text ="Angry")
    newtree1.heading("Suprised", text ="Suprised")
    newtree1.heading("Normal", text ="Normal")

    refreshproductname()

    opencvButton = Tk.Button(master=page1, text='Save Daily Record', command=_savedaily)
    opencvButton.pack(side=Tk.TOP)

    trainButton = Tk.Button(master=page1, text='Train Data', command=_opencv)
    trainButton.pack(side=Tk.TOP)


    c.execute('SELECT PRODUCT_CODE FROM PRODUCT')
    productv=[]
    for row in c.fetchall():
        productv.append(row[0])
    cbs = ttk.Combobox(master=page1,state="readonly",values=productv)
    cbs.pack(side=Tk.TOP)

    startButton = Tk.Button(master=page1, text='Start Tracking', command=lambda:_new(cbs))
    startButton.pack(side=Tk.TOP)

    #Page2
    productb =[]
    c.execute('SELECT DISTINCT product_brand FROM PRODUCT')
    for row in c.fetchall():
        productb.append(row[0])

    cb1 = ttk.Combobox(master=page2,state="readonly",values=productb)
    cb1.pack(side=Tk.TOP)

    barButton = Tk.Button(master=page2, text='Bar Chart', command=lambda:_barchart(cb1,productb))
    barButton.pack(side=Tk.TOP)

    fig2 = Figure(figsize=(5,4),dpi=100)
    a2 = fig2.add_subplot(111)

    canvas2 = FigureCanvasTkAgg(fig2,master = page2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=Tk.BOTTOM,fill=Tk.BOTH,expand=True)

    toolbar2 = NavigationToolbar2Tk(canvas2, page2)
    toolbar2.update()
    canvas2._tkcanvas.pack(side=Tk.BOTTOM,fill=Tk.BOTH,expand=True)

    #Page3
    productn =[]
    c.execute('SELECT PRODUCT_CODE FROM PRODUCT')
    for row in c.fetchall():
        productn.append(row[0])

    cb2 = ttk.Combobox(master=page3,state="readonly",values=productn)
    cb2.pack(side=Tk.TOP)

    lineButton = Tk.Button(master=page3, text='View Line Graph', command=lambda:_linegraph(cb2))
    lineButton.pack(side=Tk.TOP)

    fig = Figure(figsize=(5,4),dpi=100)
    a = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig,master = page3)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.BOTTOM,fill=Tk.BOTH,expand=True)

    toolbar = NavigationToolbar2Tk(canvas, page3)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.BOTTOM,fill=Tk.BOTH,expand=True)

    
    #Page4
    newtree = ttk.Treeview(page4)
    newtree.pack(padx=10,pady=10)

    newtree["columns"] = ["Product Code", "Product Brand", "Product Category", "Product Name"]    
    newtree["show"] = "headings"
    newtree.heading("Product Code", text ="Product Code")
    newtree.heading("Product Brand", text ="Product Brand")
    newtree.heading("Product Category", text ="Product Category")
    newtree.heading("Product Name", text ="Product Name")


    c.execute('SELECT * FROM PRODUCT')
    productdata = c.fetchall()

    index = iid = 0
    for row in productdata:
        newtree.insert("", index, iid, values = row)
        index = iid = index + 1


    addProductButton = Tk.Button(master=page4, text='Add Product', command=lambda:_product(newtree))
    addProductButton.pack(side=Tk.TOP)

    editProductButton = Tk.Button(master=page4, text='Edit Product', command=lambda:editp(newtree))
    editProductButton.pack(side=Tk.TOP)

    deleteProductButton = Tk.Button(master=page4, text='Delete Product', command=lambda:deletep(newtree))
    deleteProductButton.pack(side=Tk.TOP)

    #authorVar = Tk.StringVar()
    #authorLabel = Tk.Label(master=page1, textvariable=authorVar)
    #authorString = "\n\n Developed By: " \
    #               "\n Emotion Tracking Team " \
    #               "\n [FYP - Swinburne]"     # Initial print
    #authorVar.set(authorString)
    #authorLabel.pack(side=Tk.BOTTOM)

    #root.iconbitmap(r'..\icon\happy-sad.ico')
    
    
    Tk.mainloop()                               # Starts mainloop required by Tk


    