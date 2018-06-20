""" based on http://blog.christianperone.com/2016/01/convolutional-hypercolumns-in-python/"""
from matplotlib import pyplot as plt

import cv2
import numpy as np
import scipy as sp

import keras
from keras import Model

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K

from CityscapesHandler import CityscapesHandler
from ExtractHypercolumnLayer import ExtractHypercolumnLayer


def get_minibatches():
    """this function chooses the next minibatch"""
    return

def sample_pixels():
    """this function samples the pixel"""
    return
    
    
def build_model(num_classes, pretrained=True):
    #compile model
    if(pretrained):
        model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
    else:
        model = keras.applications.vgg16.VGG16(include_top=False, weights="none")
        
    # model_final = Sequential()
    # model_final.add(model)
    # model_final.add(ExtractHypercolumnLayer(model))
    
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model_final.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return model
    
    
def train_model(model, x, y, batch_size=1, epochs=50):
    model.fit(x, y, batch_size, epochs)
    

def predict_model(model, x):
    model.predict(x)


def get_conv2d_outputs(model, instance):  
    
    # must be adjusted to our purposes
    conv2d_layer_ids = [1, 2, 4, 5, 7, 8,  9, 11, 12, 13, 15, 16, 17]
    
    conv2d_layers = []   
    for id in conv2d_layer_ids:
        conv2d_layers.append(model.layers[id])
    
    inp = model.input                     
    outputs = [layer.output for layer in conv2d_layers]
    
    #print(K.learning_phase())
    functor = K.function([inp]+ [K.learning_phase()], outputs )
    layer_outs = functor([[instance], 1.])
            
    return layer_outs

    
def extract_hypercolumn(model, instance):
    feature_maps = get_conv2d_outputs(model, instance)
    
    #print(feature_maps)
    hypercolumns = []
    for idx, convmap in enumerate(feature_maps):   
        convmap = np.rollaxis(convmap, 3, 1)
        for fmap in convmap[0]:
            scaled = sp.misc.imresize(fmap, size=instance.shape,
                                        mode="F", interp='bilinear')
            hypercolumns.append(scaled)
    
    hypc = np.asarray(hypercolumns)
    print('shape of hypercolumns ', hypc.shape)
    return hypc

    
def main():

    csh = CityscapesHandler()
    
    train_x, train_y = csh.getTrainSet(200)
    # test_x, test_y = csh.getTestSet(5)
    # val_x, val_y = csh.getValSet(5)
    
    print(train_x.shape)
    #csh.displayImage(train_x[0])
    

    
    model = build_model(10)
    predict_model(model, np.array([train_x[0]]))
    
    hc = extract_hypercolumn(model, train_x[0])
    ave = np.average(hc.transpose(1, 2, 0), axis=2)
    print("ave", ave.shape)
    print(ave)
    plt.imshow(ave)
    plt.show()
    
if __name__ == "__main__":
    main()