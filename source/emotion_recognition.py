
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
from subprocess import Popen
import sklearn
import datetime
from sklearn import datasets
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
import cv2
if sys.version_info[0] < 3:
    import Tkinter as Tk
    import ttk

else:
    import tkinter as Tk
    from tkinter import ttk
    from tkinter import filedialog
    from tkinter.ttk import *
from PIL import ImageTk
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

def _barcata(combo2):
    ids =[]
    timesd = []
    happy = []
    total = 0
    sumhappy = 0
    subtotal = 0
    a.clear()
    global countx 
    countx = 0
    searchdb = 'SELECT DISTINCT product_brand FROM PRODUCT WHERE product_category = ?'
    c.execute(searchdb,[combo2.get()])

    for rowss in c.fetchall():
        sumhappy = 0
        searchdb = 'SELECT * FROM PRODUCT WHERE product_brand = ? AND product_category = ?'
        c.execute(searchdb,[rowss[0], combo2.get()])

        for rows in c.fetchall():
            dailyrecord2 = 'SELECT * FROM EMOTION WHERE PRODUCT_CODE = ?'
            c.execute(dailyrecord2,[rows[0]])

            for row in c.fetchall():
                sumhappy = sumhappy + row[2]
                total = total + row[2] + row[3] + row[4] + row[5] + row[6]
        
        
        subtotal = ((float(sumhappy) / float(total) * 100.00))
        countx = countx + 1 
        ids.append(rowss[0])
        happy.append(subtotal)
        

    if countx == 1 :
        a.set_xlim(-0.1,1)  
    countx = np.arange(countx)

    a.bar(countx,happy,0.10, alpha=0.8,color='green',label='Happy')
    

    a.set_xlabel('Product')
    a.set_ylabel('% of Hapiness over Total Emotion of the Brand')
    a.set_xticks(countx)
    a.set_xticklabels(ids)
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

    sumhappy = 0
    sumsad = 0
    sumangry = 0
    sumsuprised = 0
    sumnormal = 0

    a2.clear()
    global countx 
    countx = 0
    searchdb = 'SELECT * FROM PRODUCT WHERE product_brand = ?'
    c.execute(searchdb,[combo1.get()])

    for row in c.fetchall():
        sumhappy = 0
        sumsad = 0
        sumangry = 0
        sumsuprised = 0
        sumnormal = 0
        dailyrecord2 = 'SELECT * FROM EMOTION WHERE PRODUCT_CODE = ?'
        c.execute(dailyrecord2,[row[0]])

        for row in c.fetchall():
            
            sumhappy = sumhappy + row[2]
            sumsad = sumsad + row[3]
            sumangry = sumangry + row[4]
            sumsuprised = sumsuprised + row[5]
            sumnormal = sumnormal + row[5]
        ids.append(row[0])
        timesd.append(row[1])
        happy.append(sumhappy)
        sad.append(sumsad)
        angry.append(sumangry)
        suprised.append(sumsuprised)
        normal.append(sumnormal)    

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

def _product(nb, canvas,canvas2, page, cb1, cb2, cbs):
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
    refreshcombo(nb, canvas,canvas2,cb1,cb2, cbs)

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

def deletep(nb, canvas,canvas2,tree, cb1, cb2, cbs):
    for selected_item in tree.selection():
        c.execute('DELETE from PRODUCT WHERE PRODUCT_CODE=?',(tree.set(selected_item,'#1'),))

        db.commit()
        refreshcombo(nb, canvas,canvas2,cb1,cb2, cbs)
        tree.delete(selected_item)

def deleterecord(tree):
    delcode = ''
    deldate = ''
    for selected_item in tree.selection(): 
        c.execute('SELECT * from EMOTION WHERE PRODUCT_CODE= ?',(tree.set(selected_item,'#1'),))
        for row in c.fetchall():
            delcode = row[0]
            deldate = row[1]
        delerecord = 'DELETE from EMOTION WHERE PRODUCT_CODE = ? AND DATES = ?'
        c.execute(delerecord, [delcode,deldate])
        db.commit()
        tree.delete(selected_item)

def editp(nb, canvas,canvas2,tree, cb1, cb2, cbs):

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
        refreshcombo(nb, canvas,canvas2,cb1,cb2, cbs)


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


def newdata():
    datetoday = datetime.datetime.now().date()

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

def addnew():
    tempvalidation = False
    datetoday = datetime.datetime.now().date()
    checking = 'SELECT DISTINCT PRODUCT_CODE FROM (SELECT DISTINCT PRODUCT_CODE FROM EMOTION WHERE DATES = ? UNION ALL SELECT DISTINCT PRODUCT_CODE FROM TEMP) tbl GROUP BY PRODUCT_CODE HAVING COUNT(*) = 1'
    c.execute(checking, [datetoday])
    for emotion_code in c.fetchall():
        thappy = 0
        tsad = 0
        tangry = 0
        tsuprised = 0
        tnormal = 0
        saved = 'SELECT * FROM TEMP WHERE PRODUCT_CODE = ?'
        c.execute(saved, [emotion_code[0]])
        for row in c.fetchall():
            if row[0] == emotion_code[0]:
                thappy = thappy + row[2]
                tsad = tsad + row[3]
                tangry = tangry + row[4]
                tsuprised = tsuprised + row[5]
                tnormal = tnormal + row[6]
                tempvalidation = True

        if tempvalidation == True:
            savedb = 'INSERT INTO EMOTION(PRODUCT_CODE, DATES, daily_happy, daily_sad, daily_angry, daily_suprised, daily_normal) VALUES (?,?,?,?,?,?,?)'
            c.execute(savedb,[emotion_code[0],datetoday,thappy,tsad,tangry,tsuprised,tnormal])
            tempvalidation = False

def _savedaily():
    global thappy
    global tsad
    global tangry
    global tsuprised
    global tnormal
    global tname
    thappy = 0
    tsad = 0
    tangry = 0
    tsuprised = 0
    tnormal = 0
    row = []
    valid = False
    validate = False
    datetoday = datetime.datetime.now().date()

    checkdate = 'SELECT * FROM EMOTION'
    c.execute(checkdate)    
    for row in c.fetchall():
        if str(row[1]) == str(datetoday):
            valid = True
        else:
            valid = False    




    checkdate1 = 'SELECT * FROM EMOTION'
    c.execute(checkdate1)    
    row = c.fetchone()
    if not row:
        newdata()

    else:
        checkdate = 'SELECT * FROM EMOTION'
        c.execute(checkdate)    
        for row in c.fetchall():
        
            thappy = 0
            tsad = 0
            tangry = 0
            tsuprised = 0
            tnormal = 0
            if str(row[1]) == str(datetoday):
                checkid = 'SELECT * FROM TEMP'
                c.execute(checkid)
                for temprow in c.fetchall():
                    if temprow[0] == row[0]:

                        thappy = thappy + temprow[2]
                        tsad = tsad + temprow[3]
                        tangry = tangry + temprow[4]
                        tsuprised = tsuprised + temprow[5]
                        tnormal = tnormal + temprow[6]
                    else:
                        if validate == False:
                            addnew()
                            validate = True

                thappy = thappy + row[2]
                tsad = tsad + row[3]
                tangry = tangry + row[4]
                tsuprised = tsuprised + row[5]
                tnormal = tnormal + row[6]



                update1 = 'UPDATE EMOTION SET daily_happy = ? WHERE PRODUCT_CODE = ? AND DATES = ?'
                c.execute(update1,[thappy,row[0],datetoday])
                update2 = 'UPDATE EMOTION SET daily_sad = ? WHERE PRODUCT_CODE = ? AND DATES = ?'
                c.execute(update2,[tsad,row[0],datetoday])
                update3 = 'UPDATE EMOTION SET daily_angry = ? WHERE PRODUCT_CODE = ? AND DATES = ?'
                c.execute(update3,[tangry,row[0],datetoday])
                update4 = 'UPDATE EMOTION SET daily_suprised = ? WHERE PRODUCT_CODE = ? AND DATES = ?'
                c.execute(update4,[tsuprised,row[0],datetoday])
                update5 = 'UPDATE EMOTION SET daily_normal = ? WHERE PRODUCT_CODE = ? AND DATES = ?'
                c.execute(update5,[tnormal,row[0],datetoday])
                db.commit()

            else:
                if valid == False:
                    newdata()
                    valid = True    
             



    for i in newtree1.get_children():
            newtree1.delete(i)


    c.execute('SELECT * FROM EMOTION')
    productdata1 = c.fetchall()
    index1 = iid2 = 0
    for row1 in productdata1:
        newtree1.insert("", index1, iid2, values = row1)
        index1 = iid2 = index1 + 1

    delrecord = 'DELETE FROM TEMP'
    c.execute(delrecord)
    db.commit()

def _new(combo, choice, cameracb):
    #, camnum, video_feed, read_value, webcam_image, videoname
    if cameracb is not None:

        import train_classifier_videofeed
        startto = Process(target = lambda:train_classifier_videofeed.main(choice.get(), int(cameracb.get()), video_name[int(cameracb.get())]))
        
        startto.start()
        #camerastatus(cameracb.get()) = True
        #Popen('python realtime.py', shell='true')
        

def refreshcombo(nb, canvas,canvas2,cb1, cb2, cbs):
    productb2 = []
    productn2 = []
    productv2 = []
    cb1.config(values='')
    cb2.config(values='')
    cbs.config(values = '')
    c.execute('SELECT DISTINCT product_brand FROM PRODUCT')
    for row in c.fetchall():
        productb2.append(row[0])
    cb1.config(values = productb2)

    c.execute('SELECT DISTINCT product_category FROM PRODUCT')
    for row in c.fetchall():
        productn2.append(row[0])
    cb2.config(values = productn2)

    c.execute('SELECT DISTINCT PRODUCT_CODE FROM PRODUCT')
    for row in c.fetchall():
        productv2.append(row[0])
    cbs.config(values = productv2)

#detect cameras
def findcamera():
    validate = 0
    cameracount = []
    camerastatus = []
    videofeed = []
    readvalue = []
    webcamimage = []
    video_name=[]
    while(True):
        cap = cv2.VideoCapture(validate)
        if cap is None or not cap.isOpened():
            cameracount.append("---None---")

            break
        cameracount.append(validate)
        
        tvideo_name = 'Camera %s' % (validate)
        video_name.append(tvideo_name)
        validate = validate + 1   
        cap.release()

    return cameracount, video_name

def show_line():
    Popen('python realtime.py', shell='true')


def refreshcameracb(cameracombo):
    global cameracount
    global camerastatus
    global videofeed
    global readvalue
    global webcamimage
    global video_name
    cameracombo.config(value = '')

    cameracount, video_name = findcamera()
   
    cameracombo.config(value = cameracount)


if __name__ == "__main__":
   with sqlite3.connect('database.db') as db:
    c=db.cursor()



    product1=[]
    product2=[]
    product3=[]
    product4=[]
    cameracount = []
    camerastatus = []
    videofeed = []
    readvalue = []
    webcamimage = []
    video_name=[]

    # Embedding things in a tkinter plot & Starting tkinter plot
    
    matplotlib.use('TkAgg')
    root = Tk.Tk()
    root.wm_title("Emotion Analytics System")
    root.geometry('900x700')


    root.protocol('WM_DELETE_WINDOW',_quit)




    rows = 0
    while rows < 50:
        root.rowconfigure(rows, weight=1)
        root.columnconfigure(rows, weight=1)
        rows +=1

    nb = ttk.Notebook(root)
    nb.grid(row=1,column=0, columnspan=50, rowspan=49,sticky='NEWS')

    

    page1 = ttk.Frame(nb)
    nb.add(page1,text='Record')

    page2 = ttk.Frame(nb)
    nb.add(page2,text='Brand')

    page3 = ttk.Frame(nb)
    nb.add(page3,text='Category')
   
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

    c.execute('SELECT * FROM EMOTION')
    productdata1 = c.fetchall()
    index1 = iid2 = 0
    for row in productdata1:
        newtree1.insert("", index1, iid2, values = row)
        index1 = iid2 = index1 + 1

    opencvButton = Tk.Button(master=page1, text='Save Daily Record', command=_savedaily)
    opencvButton.pack(side=Tk.TOP)

    deleteRecordButton = Tk.Button(master=page1, text='Delete Record', command=lambda:deleterecord(newtree1))
    deleteRecordButton.pack(side=Tk.TOP)
    #trainButton = Tk.Button(master=page1, text='Train Data', command=_mains)
    #trainButton.pack(side=Tk.TOP)
    showline = Tk.Label(master = page1, text = "Start Real-time Line Graph: ")
    showline.pack(side = Tk.TOP)
    lineButton = Tk.Button(master=page1, text='Show Line', command=show_line)
    lineButton.pack(side=Tk.TOP)


    labelcam = Tk.Label(master = page1, text = "Camera Number: ")
    labelcam.pack(side = Tk.TOP)
    cameracount, video_name = findcamera()
    cameracb = ttk.Combobox(master=page1,state="readonly",values= cameracount)
    cameracb.set(cameracount[0])
    cameracb.pack(side=Tk.TOP)

    refcamButton = Tk.Button(master=page1, text='Refresh Camera', command=lambda:refreshcameracb(cameracb))
    refcamButton.pack(side=Tk.TOP)


    productlabel = Tk.Label(master = page1, text = "Select a Product: ")
    productlabel.pack(side = Tk.TOP)
    c.execute('SELECT PRODUCT_CODE FROM PRODUCT')
    productv=[]
    for row in c.fetchall():
        productv.append(row[0])
    cbs = ttk.Combobox(master=page1,state="readonly",values=productv)
    cbs.pack(side=Tk.TOP)

    startButton = Tk.Button(master=page1, text='Start Tracking', command=lambda:_new(cb2, cbs, cameracb))
    startButton.pack(side=Tk.TOP)

    exitButton = Tk.Button(master=page1, text='Exit', command=_quit)
    exitButton.pack(side=Tk.BOTTOM)

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
    c.execute('SELECT DISTINCT product_category FROM PRODUCT')
    for row in c.fetchall():
        productn.append(row[0])

    cb2 = ttk.Combobox(master=page3,state="readonly",values=productn)
    cb2.pack(side=Tk.TOP)

    lineButton = Tk.Button(master=page3, text='View Bar Chart', command=lambda:_barcata(cb2))
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


    addProductButton = Tk.Button(master=page4, text='Add Product', command=lambda:_product(nb, canvas,canvas2,newtree, cb1, cb2, cbs))
    addProductButton.pack(side=Tk.TOP)

    editProductButton = Tk.Button(master=page4, text='Edit Product', command=lambda:editp(nb, canvas,canvas2,newtree, cb1, cb2, cbs))
    editProductButton.pack(side=Tk.TOP)

    deleteProductButton = Tk.Button(master=page4, text='Delete Product', command=lambda:deletep(nb, canvas,canvas2,newtree, cb1, cb2, cbs))
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


    
