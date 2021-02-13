from face_detector.detector import Detector
from tensorflow import keras
from flask import Response
from flask import Flask
from flask import render_template
from imutils.video import VideoStream
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe exchanges of the output frames (useful when multiple browsers/tabs are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

#path of all models
prototxtpath = './face_detector/face_model/deploy.prototxt'
caffemodelpath = './face_detector/face_model/res10_300x300_ssd_iter_140000.caffemodel'
modelpath = './face_detector/mask_model/model.h5'

#models
facenet = cv2.dnn.readNet(prototxtpath, caffemodelpath)
masknet = keras.models.load_model(modelpath)

#start video stream
cam = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    return render_template('index.html')

def mask_detector():
	#grab global references to the video stream, model, output frame, and lock variables
    global cam, facenet, masknet, outputFrame, lock

    #initialize detector
    detector = Detector()

    #stream
    while True:
        frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=680)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 225, 255), 1)
        (locs, preds) = detector.detect_and_predict(frame, facenet, masknet)

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            (X, Y, eX, eY) = box
            (mask, withoutmask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutmask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            #label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (X, Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (X, Y), (eX, eY), color, 2)

        with lock:
            outputFrame = frame.copy()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media type (mime type)
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=mask_detector)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

# release the video stream pointer
cam.stop()