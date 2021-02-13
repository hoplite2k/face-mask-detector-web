from tensorflow import keras
import numpy as np
import cv2
import imutils
import time
import h5py
from imutils.video import VideoStream

class Detector:

	def detect_and_predict(self, frame, facenet, masknet):
	    (h, w) = frame.shape[:2]
	    # create a blob for frame
	    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	    # detect faces
	    facenet.setInput(blob)
	    detections = facenet.forward()

	    # initialize the list of faces, their corresponding locations and the list of predictions from our face mask network
	    faces = []
	    locs = []
	    preds = []

	    # loop over detections
	    for i in range(detections.shape[2]):
	        confidence = detections[0, 0, i, 2]
	        if confidence > 0.5:
	            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	            (X, Y, eX, eY) = box.astype('int')
	            # ensure whether box is inside frame
	            (X, Y) = (max(0, X), max(0, Y))
	            (eX, eY) = (min(w - 1, eX), min(h - 1, eY))

	            # extract face
	            face = frame[Y:eY, X:eX]
	            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	            face = cv2.resize(face, (224, 224))
	            face = keras.preprocessing.image.img_to_array(face)
	            face = keras.applications.mobilenet_v2.preprocess_input(face)
	            face = np.expand_dims(face, axis=0)

	            # append face and its location
	            faces.append(face)
	            locs.append((X, Y, eX, eY))

	    # prediction
	    if len(faces) > 0:
	        preds = masknet.predict(faces)

	    return (locs, preds)
