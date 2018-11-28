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
from face_detection import find_faces


svc_1 = SVC(kernel='linear')  # Initializing Classifier
svc_2 = SVC(kernel='linear')
svc_3 = SVC(kernel='linear')
svc_4 = SVC(kernel='linear')
svc_5 = SVC(kernel='linear')

global last_frame
last_frame = np.zeros((480,640,3),dtype=np.uint8)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)

#faces = datasets.fetch_olivetti_faces()
# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================
class Trainer:


    
    def __init__(self):
        
        self.index = 0

    def reset(self):
        
        self.index = 0

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


# Trained classifier's performance evaluation
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    #scores = cross_val_score(clf, X, y, cv=cv)
    #print ("Scores: ", (scores))
    #print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# Confusion Matrix and Results
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    #print ("Accuracy on training set:")
    #print (clf.score(X_train, y_train))
    #print ("Accuracy on testing set:")
    #print (clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    #print ("Classification Report:")
    #print (metrics.classification_report(y_test, y_pred))
    #print ("Confusion Matrix:")
    #print (metrics.confusion_matrix(y_test, y_pred))


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
    new_extracted_face = zoom(extracted_face, (62. / extracted_face.shape[0],
                                               47. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face


# ------------------- LIVE FACE RECOGNITION -----------------------------------




def main(lmain, camnum, videoname):
    global happy
    global sad
    global total
    global angry
    global suprised
    global normal
    global count
    global t
    global displaytext
    global timecount
    global check

    happy = 0.00
    sad = 0.00
    angry = 0.00
    suprised = 0.00
    normal = 0.00
    happydb = 0.00
    saddb = 0.00
    angrydb = 0.00
    supriseddb = 0.00
    normaldb = 0.00
    total = 0.00
    t=0
    timecount = 0
    timecountsecond = 0
    timecountminutes = 0
    check = False

    emotions = ["happy", "sad", "angry", "surprised", "normal"]

    # Load model
    fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emotion.read('models/emotion_classifier_model.xml')

    
   
    count = 0

    trainer = Trainer()


    video_feed = cv2.VideoCapture(camnum)
    
    read_value, webcam_image = video_feed.read()


    init = True
    while read_value:
        face_index = 0
        timecountsecond = 0
        now = datetime.datetime.now()
        timecountsecond = now.second

        if timecount == 0:
            timecount = timecountsecond

        if timecountsecond == 0:
            timecount = 0

        if timecount == 58:
            timecount = -2

        elif timecount == 59:
            timecount = -1

        if timecount + 2 == timecountsecond:
            timecount = timecountsecond
            total = total + 1
            check = True

        read_value, webcam_image = video_feed.read()

        
            
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            
            emotion_prediction = fisher_face_emotion.predict(normalized_face)
            cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (255,0,0), 2)
            #cv2.putText(webcam_image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
            
            if emotion_prediction[0] == 0:
                happy = happy + 1

            elif emotion_prediction[0] == 1:
                sad = sad + 1

            elif emotion_prediction[0] == 2:
                angry = angry + 1

            elif emotion_prediction[0] == 3:  
                suprised = suprised + 1

            elif emotion_prediction[0] == 4:
                normal = normal + 1 


            if check == True:
                check = False

                if happy >= sad and happy >=angry and happy >=suprised and happy >=normal:
                    cv2.putText(webcam_image, "HAPPY", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
                    happydb = happydb + 1


                elif sad >= happy and sad >=angry and sad >=suprised and sad >=normal:
                    cv2.putText(webcam_image, "SAD", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2) 
                    saddb = saddb + 1

                elif angry >= happy and angry >=sad and angry >=suprised and angry >=normal:
                    cv2.putText(webcam_image, "ANGRY", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
                    angrydb = angrydb + 1

                elif suprised >= happy and suprised >= angry and suprised >= sad and suprised >=normal:
                    cv2.putText(webcam_image, "SURPRISED", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
                    supriseddb = supriseddb + 1

                else:   
                    cv2.putText(webcam_image, "NORMAL", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2) 
                    normaldb = normaldb + 1

                
                happy = 0.00
                sad = 0.00
                angry = 0.00
                suprised = 0.00
                normal = 0.00

        if total == 29:
            timecountminutes = timecountminutes + 1
            total = 0


            # increment counter
            face_index += 1
            count = count + 1

        if timecountminutes == 1:    
                    
                
            t = now.strftime("%H:%M:%S")

            with sqlite3.connect('database.db') as db:
                c = db.cursor()

            savetemp = 'INSERT INTO TEMP(PRODUCT_CODE,times,happy,sad,angry,suprised,normal) VALUES (?,?,?,?,?,?,?)'
            c.execute(savetemp,[lmain,t,happydb,saddb,angrydb,supriseddb,normaldb])

            db.commit()
            timecountminutes = 0
            count = 0.00
            happy = 0.00
            sad = 0.00
            angry = 0.00
            suprised = 0.00
            normal = 0.00
            happydb = 0.00
            saddb = 0.00
            angrydb = 0.00
            supriseddb = 0.00
            normaldb = 0.00

        # Display the resulting frame
        cv2.imshow(videoname, webcam_image)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    video_feed.release()
    cv2.destroyAllWindows()

#if __name__ == "__main__":
#    main("Product_1")

    # When everything is done, release the capture
    
