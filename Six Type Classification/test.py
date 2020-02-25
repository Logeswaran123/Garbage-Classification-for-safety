# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:31:28 2020

@author: admin
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import time
import cvlib as cv

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Read video
#video = cv2.VideoCapture("videoplayback.mp4")
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (1920, 1080))

model_path = 'keras_classification_model.h5'
# load pre-trained model
model = load_model(model_path)

# Exit if video not opened
#if not video.isOpened():
 #   print("Could not open video")

# Read first frame
#ok, frame = video.read()
#if not ok:
 #   print('Cannot read video file')

while True:
    
    
    start_time = time.time()
    frame_ = cv2.imread('paper.jpg')
    
    #frame = cv2.imread("1.jpg")
    (H, W) = frame_.shape[:2]
    print(H, W)
    
    
    
    classes = ['cardboard','glass', 'metal', 'paper', 'plastic', 'trash']
    


    # preprocessing for gender detection model
    frame = cv2.resize(frame_, (300,300))
    frame = frame.astype("float") / 255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)

    # apply gender detection on face
    predictions = model.predict(frame)
    print(predictions)
    conf = predictions[0]
    print(conf)
    print(classes)
    
    
    # get label with max accuracy
    idx = np.argmax(conf)
    label = classes[idx]

    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

    Y = 100 - 10 if 100 - 10 > 10 else 100 + 10

    # write label and confidence above face rectangle
    cv2.putText(frame_, label, (150, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 1)
    # display output
    cv2.imshow("gender detection", frame_)
    #out.write(frame)
    # Break if ESC pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop



# save output
#cv2.imwrite("gender_detection.jpg", image)
#out.release()
#video.release()
# release resources
cv2.destroyAllWindows()