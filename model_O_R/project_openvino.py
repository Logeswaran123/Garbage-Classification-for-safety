# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:19:14 2020

@author: admin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import itertools
import os
import pickle
from random import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Activation,Dropout
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_files
from smallervggnet import SmallerVGGNet



# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64

path = 'DATASET/DATASET/TRAIN/allimgs/'
files = os.listdir(path)
print(len(files))

shuffle(files)
gender = [i.split('_')[0] for i in files]
print(len(gender))


classes = []
for i in gender:
    if i == 'O':
        classes.append(0)
    else:
        classes.append(1)
print(len(classes))

X_data =[]
for file in files:
    img = cv2.imread('DATASET/DATASET/TRAIN/allimgs/' + file)
    img = cv2.resize(img, (96, 96))
    X_data.append(img)

X = np.squeeze(X_data)
print(X.shape)

# normalize data
X = X.astype('float32')
X /= 255

print(classes[:10])
classes = np.array(classes)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(X, classes, test_size=0.2,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# build model
model = SmallerVGGNet.build(width = 96, height = 96, depth = 3,
                            classes = 2)

# compile the model
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint = ModelCheckpoint(filepath = 'cnn.model', verbose = 1, save_best_only = True)

earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 15, #Number of epochs we wait before stopping 
                          verbose = 1,
                          restore_best_weights = True) #keeps the best weigths once stopped

ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)

callbacks = [earlystop, checkpoint, ReduceLR]

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1, callbacks = callbacks)

# save the model to disk
model.save('keras_model.h5')

# Save History file
pickle_out = open("Trained_cnn_history.pickle","wb")
pickle.dump(H.history, pickle_out)
pickle_out.close()

# Plot training & validation accuracy values
plt.subplot(2, 2, 1)
plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')

# Plot training & validation loss values
plt.subplot(2, 2, 2)
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')


model = load_model('cnn.model')
model.load_weights('cnn.model')

score = model.evaluate(testX, testY,verbose=0)
print('Test Loss :',score[0])
print('Test Accuracy :',score[1])

#get the predictions for the test data
predicted_classes = model.predict_classes(testX)
rounded_labels = np.argmax(testY, axis=1)

confusion_mtx = confusion_matrix(rounded_labels, predicted_classes) 

plt.subplot(2, 2, 3)
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['R','O'], rotation=90)
plt.yticks(tick_marks, ['R','O'])
#Following is to mention the predicated numbers in the plot and highligh the numbers the most predicted number for particular label
thresh = confusion_mtx.max() / 2.
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, confusion_mtx[i, j],
    horizontalalignment="center",
    color="white" if confusion_mtx[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()