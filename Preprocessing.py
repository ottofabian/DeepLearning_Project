import json 
import os
import io

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from pprint import pprint
from shutil import copyfile

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

WIDTH = 224
HEIGHT = 224
as_grey = False
batch_size = 16
training_size = 0
epochs = 2

annotation_path_train = os.getcwd() + '/data/annotations/train_annotations.json'
annotation_path_val = os.getcwd() + '/data/annotations/val_annotations.json'

def img_to_folder():

  # load labels
  train_dict = {}
  train_labels=open(annotation_path_train)
  data_train = json.load(train_labels)
  for elem in data_train['annotations']:
      train_dict[elem['image_id']] = elem['category_id']

  train_labels.close()
  training_size = len(train_dict.values())

  val_dict = {}
  val_labels=open(annotation_path_val)
  data_val = json.load(val_labels)
  for elem in data_val['annotations']:
      val_dict[elem['image_id']] = elem['category_id']    

  val_labels.close()
  val_size = len(val_dict.values())  

  for key, value in train_dict.items():
      copyfile(os.getcwd() + '/data/train_val/{}.jpg'.format(key), os.getcwd() + '/data/train/{}/{}.jpg'.format(value, key))

  for key, value in val_dict.items():
      copyfile(os.getcwd() + '/data/train_val/{}.jpg'.format(key), os.getcwd() + '/data/val/{}/{}.jpg'.format(value, key))

def get_gen():
  # load images with generators
  images = []

  # augmentation configuration for training
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)

  # augmentation configuration for testing:
  val_datagen = ImageDataGenerator(rescale=1./255)

  # train batches of augmented image data
  train_generator = val_datagen.flow_from_directory(
          os.getcwd() + '/data/train',
          target_size=(HEIGHT, WIDTH),
          batch_size=batch_size,
          class_mode='categorical')

  # this is a similar generator, for validation data
  validation_generator = test_datagen.flow_from_directory(
          os.getcwd() + '/data/val',
          target_size=(HEIGHT, WIDTH),
          batch_size=batch_size,
          class_mode='categorical')
   
  test_datagen = ImageDataGenerator(rescale=1./255)

  test_generator = test_datagen.flow_from_directory(
          os.getcwd() + '/data/test',
          target_size=(HEIGHT, WIDTH),
          batch_size=40,
          class_mode=None,  # only data, no labels
          shuffle=False)

  # for image_path in os.listdir(os.getcwd() + '/data/train_val'):
  #     print(image_path)
  #     img = io.imread(input_path, as_grey=as_grey)
  #     img = img.reshape([WIDTH, HEIGHT, 1 if as_grey else 3])
  #     all_images.append(img)
  
  return train_generator, validation_generator, test_generator

def visualize(history):
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
def save_pred(pred):
  ids = [file[4:-4] for file in test_generator.filenames]
  df = pd.DataFrame(np.append(ids, pred).reshape(2, len(pred)).T)
  df.columns = ['id','animal_present']
  df.to_csv('./predictions.csv', index=False)
  
  
input_tensor = Input(shape=(WIDTH, HEIGHT, 3))

# VGG without FC layers
# model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
# model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
#                                      input_tensor=input_tensor, classes=2)
model = keras.applications.resnet50.ResNet50(include_top=False, weights=None,
                                     input_tensor=input_tensor, classes=2)
x = model(input_tensor)

# Classification block
x = Flatten(name='flatten')(x)
x = Dense(512, activation='relu', name='fc1')(x)
# x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)
model = Model(input=input_tensor, output=x)

# set up tensorboard for progress and analysis 
# tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(datetime.now()), histogram_freq=0,
#                                           batch_size=batch_size, write_graph=True, write_grads=True,
#                                           write_images=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="./model_checkpoints/{}.hdf5".format(datetime.now()), verbose=1, save_best_only=True)
history = model.fit_generator(train_generator, steps_per_epoch=training_size / batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=val_size / batch_size, callbacks=[checkpointer])

model.save(os.getcwd() + '/models/{}.h5'.format(datetime.now()))
visualize(history)
