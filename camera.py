from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2  
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import tensorflow as tf

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
#             face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces)
        preds = maskNet.predict(faces)
    return (locs, preds)

prototxtPath = r"C://Users//Admin//Untitled Folder//Tracking//Face-Mask-Detection-master//face_detector//deploy.prototxt"
weightsPath = r"C://Users//Admin//Untitled Folder//Tracking//Face-Mask-Detection-master//face_detector//res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("C://Users//Admin//Downloads//model3classFaceMarkssFaceMark224.h5")
# maskNet = load_model("C://Users//Admin//Untitled Folder//Tracking//Face-Mask-Detection-master//mask_detector.model")


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        count = int(0)
        while True:
            ret, frame = self.video.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=900)
        #     ret,frame = vs.read()
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                pred = tf.nn.softmax(pred)
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                labell = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if labell == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(labell, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                if (labell=="No Mask"):
                    cv2.imwrite(os.path.join('C://Users//Admin//OneDrive//Desktop//tkpm//face//DataNoMask', "%d.jpg", % count), frame)
                    count = count + 1
                    print(count)
                ret, jpg = cv2.imencode('.jpg', frame)
            if (count > 20):
                break
        return jpg.tobytes()