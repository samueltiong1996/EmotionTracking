import cv2
import glob
import os

from face_detection import find_faces

#clearing the previous data
def remove_face_data(emotions):
    print("Removing previous processed faces...")
    for emotion in emotions:
        filelist = glob.glob("../data/emotion/%s/*" % emotion)
        for file in filelist:
            os.remove(file)
            
    print("Done!")

    #image extraction
def extract_faces(emotions):
    print("Extracting faces...")
    add = 80
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if not os.path.exists('../data/emotion'):
        os.makedirs('../data/emotion')
    for emotion in emotions:
        print("Processing %s data..." % emotion)
        images = glob.glob('../data/raw_emotion/%s/*.tiff' % emotion)
        if not os.path.exists('../data/emotion/%s' % emotion):
            os.makedirs('../data/emotion/%s' % emotion)

        for file_number, image in enumerate(images):
            frame = cv2.imread(image)
            faces = find_faces(frame)
            for face in faces:
                try:
                    cv2.imwrite("../data/emotion/%s/%s.jpg" % (emotion, add + 1), face[0])
                    add = add + 1
                except:
                    print("Error in processing %s" % image)

    print("Face extraction finished")


if __name__ == '__main__':
    emotions = ["happy", "sad", "angry", "surprised", "normal"]
#    remove_face_data(emotions)
    extract_faces(emotions)
