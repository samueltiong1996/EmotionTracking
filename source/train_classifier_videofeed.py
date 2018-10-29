import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
import cv2
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import zoom
from sklearn import datasets
import subprocess
from subprocess import Popen
import datetime
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
    import ttk
else:
    import tkinter as Tk
    from tkinter import ttk

with sqlite3.connect('database.db') as db:
    c=db.cursor()
from PIL import Image
from PIL import ImageTk


svc_1 = SVC(kernel='linear')  # Initializing Classifier
svc_2 = SVC(kernel='linear')
svc_3 = SVC(kernel='linear')
svc_4 = SVC(kernel='linear')
svc_5 = SVC(kernel='linear')

global last_frame
last_frame = np.zeros((480,640,3),dtype=np.uint8)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)

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

    def record_result(self, smile=True):
        print ("Image", self.index + 1, ":", "Happy" if smile is True else "Sad")
        self.results[str(self.index)] = smile


# Trained classifier's performance evaluation
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print ("Scores: ", (scores))
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# Confusion Matrix and Results
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))


# ===============================================================================
# from FaceDetectPredict.py
# ===============================================================================

def detectFaces(frame):
    cascPath = "../data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, detected_faces


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    extracted_face = gray[y + vertical_offset:y + h,
                     x + horizontal_offset:x - horizontal_offset + w]
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                               64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face


def predict_face_is_smiling(extracted_face):
    return svc_1.predict(extracted_face.reshape(1, -1))

def predict_face_is_sad(extracted_face):
    return svc_2.predict(extracted_face.reshape(1, -1))

def predict_face_is_angry(extracted_face):
    return svc_3.predict(extracted_face.reshape(1, -1))

def predict_face_is_suprised(extracted_face):
    return svc_4.predict(extracted_face.reshape(1, -1))    

def predict_face_is_normal(extracted_face):
    return svc_5.predict(extracted_face.reshape(1, -1))  

def test_recognition(c1, c2):
    extracted_face1 = extract_face_features(gray1, face1[0], (c1, c2))
    print(predict_face_is_smiling(extracted_face1))
    extracted_face2 = extract_face_features(gray2, face2[0], (c1, c2))
    print(predict_face_is_smiling(extracted_face2))
    cv2.imshow('gray1', extracted_face1)
    cv2.imshow('gray2', extracted_face2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# test_recognition(0.3, 0.05)

# ------------------- LIVE FACE RECOGNITION -----------------------------------




def main(lmain):
    global happy
    global sad
    global total
    global angry
    global suprised
    global normal
    global count
    global t
    global displaytext

    happy = 0.00
    sad = 0.00
    angry = 0.00
    suprised = 0.00
    normal = 0.00
    total = 0.00
    t=0


    #Popen('python realtime.py', shell='true')
   
    count = 0

    

    trainer = Trainer()
    #Happy
    resultsh = json.load(open("../results/resultsh.xml"))  # Loading the classification result
    trainer.resultsh = resultsh

    indices = [int(i) for i in trainer.resultsh]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.resultsh[i] for i in trainer.resultsh]  # Target Vector
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Trained classifier's performance evaluation
    evaluate_cross_validation(svc_1, X_train, y_train, 5)

    # Confusion Matrix and Results
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

    #Sad
    resultss = json.load(open("../results/resultss.xml"))  # Loading the classification result
    trainer.resultss = resultss

    indices = [int(i) for i in trainer.resultss]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.resultss[i] for i in trainer.resultss]  # Target Vector
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Trained classifier's performance evaluation
    evaluate_cross_validation(svc_2, X_train, y_train, 5)

    # Confusion Matrix and Results
    train_and_evaluate(svc_2, X_train, X_test, y_train, y_test)

    #Angry
    resultsa = json.load(open("../results/resultsa.xml"))  # Loading the classification result
    trainer.resultsa = resultsa

    indices = [int(i) for i in trainer.resultsa]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.resultsa[i] for i in trainer.resultsa]  # Target Vector
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Trained classifier's performance evaluation
    evaluate_cross_validation(svc_3, X_train, y_train, 5)

    # Confusion Matrix and Results
    train_and_evaluate(svc_3, X_train, X_test, y_train, y_test)


    #Surprised
    resultssp = json.load(open("../results/resultssp.xml"))  # Loading the classification result
    trainer.resultssp = resultssp

    indices = [int(i) for i in trainer.resultssp]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.resultssp[i] for i in trainer.resultssp]  # Target Vector
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Trained classifier's performance evaluation
    evaluate_cross_validation(svc_4, X_train, y_train, 5)

    # Confusion Matrix and Results
    train_and_evaluate(svc_4, X_train, X_test, y_train, y_test)


    #Normal
    resultsn = json.load(open("../results/resultsn.xml"))  # Loading the classification result
    trainer.resultsn = resultsn

    indices = [int(i) for i in trainer.resultsn]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.resultsn[i] for i in trainer.resultsn]  # Target Vector
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Trained classifier's performance evaluation
    evaluate_cross_validation(svc_5, X_train, y_train, 5)

    # Confusion Matrix and Results
    train_and_evaluate(svc_5, X_train, X_test, y_train, y_test)


    video_capture = cv2.VideoCapture(0)
    

    while (True):

        now = datetime.datetime.now()
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # detect faces
        gray, detected_faces = detectFaces(frame)
        
        face_index = 0

        cv2.putText(frame, "Press Esc to QUIT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # predict output
        for face in detected_faces:
            (x, y, w, h) = face
            if w > 100:
                # draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # extract features
                extracted_face = extract_face_features(gray, face, (0.3, 0.05)) #(0.075, 0.05)

                # predict smile
                prediction_result = predict_face_is_smiling(extracted_face)
                prediction_results = predict_face_is_sad(extracted_face)
                prediction_resulta = predict_face_is_angry(extracted_face)
                prediction_resultsp = predict_face_is_suprised(extracted_face)
                prediction_resultn = predict_face_is_normal(extracted_face)

                # draw extracted face in the top right corner
                frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)                
                # annotate main image with a label
                if prediction_result == 1:
                    #cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                    happy = happy + 1
                if prediction_results == 1:
                    #cv2.putText(frame, "SAD",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                    sad = sad + 1
                if prediction_resulta == 1:
                    #cv2.putText(frame, "ANGRY",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                    angry = angry + 1
                if prediction_resultsp == 1:
                    #cv2.putText(frame, "SUPRISED",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                    suprised = suprised + 1
                if prediction_resultn == 1:
                    #cv2.putText(frame, "NORMAL",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                    normal = normal + 1

                if happy >= sad and happy >=angry and happy >=suprised and happy >=normal:
                    cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                elif sad >= happy and sad >=angry and sad >=suprised and sad >=normal:
                    cv2.putText(frame, "SAD",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)  
                elif angry >= happy and angry >=sad and angry >=suprised and angry >=normal:
                    cv2.putText(frame, "ANGRY",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                elif suprised >= happy and suprised >= angry and suprised >= sad and suprised >=normal:
                    cv2.putText(frame, "SUPRISED",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                else:   
                    cv2.putText(frame, "NORMAL",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)

                # increment counter
                face_index += 1
                count = count + 1

            if count == 60:    
                    
                total = happy + sad

                
                t = now.strftime("%H:%M:%S")

                with sqlite3.connect('database.db') as db:
                    c = db.cursor()

                savetemp = 'INSERT INTO TEMP(PRODUCT_CODE,times,happy,sad,angry,suprised,normal) VALUES (?,?,?,?,?,?,?)'
                c.execute(savetemp,[lmain,t,happy,sad,angry,suprised,normal])

                db.commit()
  
                count = 0.00
                happy = 0.00
                sad = 0.00
                angry = 0.00
                suprised = 0.00
                normal = 0.00
        

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    

if __name__ == "__main__":


    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
