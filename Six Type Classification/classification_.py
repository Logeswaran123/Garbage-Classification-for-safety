# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:25:31 2020

@author: admin
"""

import cv2
import pickle
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D,GlobalAveragePooling2D
from keras.models  import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from keras import optimizers, losses
import random,os,glob
import matplotlib.pyplot as plt

dir_path = 'Garbage classification'

img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))

print(len(img_list))

train = ImageDataGenerator(horizontal_flip = True, vertical_flip = True,
                         validation_split = 0.1, rescale = 1./255,
                         shear_range = 0.2, zoom_range = 0.2,
                         width_shift_range = 0.1, height_shift_range = 0.1,)

test = ImageDataGenerator(rescale = 1/255, validation_split = 0.1)

train_generator = train.flow_from_directory(dir_path,
                                          target_size = (300,300),
                                          batch_size = 32,
                                          class_mode = 'categorical',
                                          subset = 'training')

valid_generator = test.flow_from_directory(dir_path,
                                        target_size = (300,300),
                                        batch_size = 32,
                                        class_mode = 'categorical',
                                        subset = 'validation')

labels = (train_generator.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)

for image_batch, label_batch in train_generator:
  break
print(image_batch.shape, label_batch.shape)


base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))
base_model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.trainable = False
model = Sequential([base_model, GlobalAveragePooling2D(), Dropout(0.20), Dense(1024, activation='relu'), Dense(6, activation='softmax')])

opt = optimizers.nadam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

batch_size = 32
epochs = 100


steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

checkpoint = ModelCheckpoint(filepath = 'cnn.model', verbose = 1, save_best_only = True)

earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 15, #Number of epochs we wait before stopping 
                          verbose = 1,
                          restore_best_weights = True) #keeps the best weigths once stopped

ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)

callbacks = [earlystop, checkpoint, ReduceLR]

# train the model
H = model.fit_generator(generator=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=valid_generator, validation_steps=validation_steps,
                              callbacks=callbacks)

# save the model to disk
model.save('keras_classification_model.h5')

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
plt.show()
