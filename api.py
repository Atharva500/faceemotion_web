from flask import Flask,request,Response,render_template
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import pandas as pd
import numpy as np
import h5py
import cv2

cap = cv2.VideoCapture('cv2.CAP_V4L2')
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

app = Flask(__name__)

def gen_frames():  
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,6)

            for (x,y,w,h) in faces:
                cv2.rectangle(gray, (x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    preds = classifier.predict(roi)[0]
                    label = class_labels[preds.argmax()]
                    label_position = (x,y)
                    gray = cv2.putText(gray,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),3)
            ret, buffer = cv2.imencode('.jpg', gray)
            frame = buffer.tobytes()            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def welcome():
    return render_template('stream.html')

@app.route('/stream_video')
def stream():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
