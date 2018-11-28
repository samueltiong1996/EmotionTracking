import cv2
import glob
import numpy as np
import random

fisher_face1 = cv2.face.FisherFaceRecognizer_create()
fisher_face2 = cv2.face.FisherFaceRecognizer_create()
fisher_face3 = cv2.face.FisherFaceRecognizer_create()
fisher_face4 = cv2.face.FisherFaceRecognizer_create()
fisher_face5 = cv2.face.FisherFaceRecognizer_create()

def get_files(emotion, training_set_size):
    files = glob.glob("../data/emotion/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]
    return training, prediction

def make_sets(emotions):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion, 0.8)

        for item in training:
            image = cv2.imread(item, 0)
            training_data.append(image)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item, 0)
            prediction_data.append(image)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer(emotions):
    training_data, training_labels, prediction_data, prediction_labels = make_sets(emotions)

    print("Size of training set is:", len(training_labels), "images")
    fisher_face1.train(training_data, np.asarray(training_labels))

    print("Size of prediction test is:", len(prediction_labels), "images")
    correct = 0
    for idx, image in enumerate(prediction_data):
        if (fisher_face1.predict(image)[0] == prediction_labels[idx]):
            correct += 1

    percentage = (correct * 100) / len(prediction_labels)

    return correct, percentage

def _mains():
    emotions = ["happy", "sad", "angry", "surprised", "normal"]

    correct, percentage = run_recognizer(emotions)
    print("Processed ", correct, " data correctly")
    print("Got ", percentage, " accuracy")
    print("Emotion training data completed")

    fisher_face1.write('models/emotion_classifier_model.xml')
