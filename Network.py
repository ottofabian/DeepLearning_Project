""" based on http://blog.christianperone.com/2016/01/convolutional-hypercolumns-in-python/"""
from skimage import transform
from matplotlib import pyplot as plt

import theano
import cv2
import numpy as np
import scipy as sp

import keras
from keras import Model

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K

from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

def get_minibatches():
    """this function chooses the next minibatch"""
    return

def sample_pixels():
    """this function samples the pixel"""
    return

#compile model
model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

#test on image
im_original = cv2.resize(cv2.imread('example_pic.jpg'), (224, 224))
#image = im_original.transpose((2,0,1))
image = np.expand_dims(im_original, axis=0)


def get_layer_outputs(model, layer_indexes, instance):
    """returns the outputs of the respective layers which are named in the layer_indexes list"""
    layer_outputs = []
    test_image = instance
    for layer in layer_indexes:
        intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer].output)
        intermediate_output = intermediate_layer_model.predict(test_image)
        layer_outputs.append(intermediate_output)
    return layer_outputs

def extract_hypercolumn(model, layer_indexes, instance):
    test_image = instance
    feature_maps = get_layer_outputs(model, layer_indexes, test_image)
    #print(feature_maps)
    hypercolumns = []
    for idx, convmap in enumerate(feature_maps):
        for fmap in convmap[0]:
            #upscaled = transform.resize(fmap, (224,224))
            #upscaled = cv2.resize(test_image, dsize=(224,224))
            upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)

    hypc = np.asarray(hypercolumns)
    print('shape of hypercolumns ', hypc.shape)
    print(hypercolumns)
    return np.asarray(hypercolumns)

# let’s see how these hypercolumns looks like for the layers 4 and 6
layers_extract = [4,6]
hc = extract_hypercolumn(model, layers_extract, image)
ave = np.average(hc.transpose(1, 2, 0), axis=2)
plt.imshow(ave)
plt.show()