from data.cityscapesscripts.helpers import csHelpers
from data.cityscapesscripts.helpers import labels

from sklearn.preprocessing import LabelEncoder

import PIL
from PIL import Image
import scipy as sp
import os
import numpy as np
from skimage import transform
from matplotlib import pyplot as plt

import cv2
import numpy as np
import scipy as sp
import tensorflow as tf
from PIL import Image

import keras
from keras import Model

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K

from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn.preprocessing import StandardScaler


x_data_root = "./data/leftImg8bit"
labels_data_root = "./data/gtFine"
default_image_shape = (320, 160)
useTrainingLabels = False

class CityscapesHandler(object):

    def getNumLabels(self):
        return len(labels.labels) - 1

        
    def getClassNameFromId(self, class_id):
        return labels.id2label[class_id].name

        
    def getClassIdFromName(self, class_name):
        return labels.name2label[class_name].id
        
        
    def getImageFromFilename(self, filename):
        return csHelpers.getCsFileInfo(filename)

        
    def getDataset(self, setType, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale = False):
        x = []
        y = []
        
        x_root = x_data_root + "/" + setType.lower() 
        y_root = labels_data_root + "/" + setType.lower()
        
        if(specificCity.lower() != "all"):
            x_root += "/" + specificCity.lower()
            y_root += "/" + specificCity.lower()
        
        counter = 0
        finished = False
        for dirName, subdirList, fileList in os.walk(x_root):
            for fname in fileList:
                img = Image.open(dirName + "/" + fname)
                
                if(asGreyScale):
                    img = img.convert("L")
                
                img = img.resize(shape)
                #x[csHelpers.getCoreImageFileName(fname)] = np.array(img)
                x.append(np.array(img))
                counter += 1
                    
                if(counter == maxNum):
                    finished = True
                    break
            if(finished):
                break
                
        counter = 0
        finished = False
        for dirName, subdirList, fileList in os.walk(y_root):
            for fname in fileList:
                
                if(fname.endswith("_gtFine_labelIds.png")):        
                    img = Image.open(dirName + "/" + fname)             
                    img = img.resize(shape)
                    #x[csHelpers.getCoreImageFileName(fname)] = np.array(img)
                    y.append(np.array(img))
                    counter += 1
                        
                    if(counter == maxNum):
                        finished = True
                        break
            if(finished):
                break
                
        print(str(counter) + " images with shape " + str(shape) + " read for " + setType + "_set.")          
        return np.array(x), np.array(y)

        
    def getTrainSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale = False):
        return self.getDataset("train", maxNum, specificCity, shape, asGreyScale)


    def getTestSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale = False):
        return self.getDataset("test", maxNum, specificCity, shape, asGreyScale)

        
    def getValSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale = False):
        return self.getDataset("val", maxNum, specificCity, shape, asGreyScale)

        
    def evaluateResults(self, predictions, groundTruths):
        pass

        
    def fromLabelIDsTo1hot(self, labels):
        numLabels = self.getNumLabels()
        result = np.zeros((len(labels), numLabels))

        for idx, e in enumerate(labels):
            result[idx][e] = 1
            
        return result

        
    def from1hotToLabelIDs(self, labels):
        return np.argmax(labels, axis=1)
     
     
    #dummy implementation
    def samplePixels(self, numSamples=2000, imageShape=default_image_shape):
        result = []
        for k in range(0, numSamples):
            x = np.random.randint(0, high=imageShape[0]-1)
            y = np.random.randint(0, high=imageShape[1]-1)
            result.append(np.array([x,y]))
        
        return np.array(result)

        
    def displayImage(self, image):
        img = Image.fromarray(image)
        img.format = "PNG"
        img.show()


def get_minibatches():
    """this function chooses the next minibatch"""
    return


def get_layer_outputs(model, layer_indexes, instance, coordinates):
    """returns the outputs of the respective layers which are named in the layer_indexes list"""
    layer_outputs = []
    #pixel = instance [coordinates[1]] [coordinates[0]]
    for layer in layer_indexes:
        intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer].output)
        intermediate_output = intermediate_layer_model.predict(instance)
        #layer_outputs.append(intermediate_output[coordinates[1]][coordinates[0]])
        layer_outputs.append(intermediate_output)
        print("Intermediate output:", intermediate_output)
    return layer_outputs


def extract_hypercolumn(model, layer_indexes, instance, coordinates):
    """hypercolumn needs to be extracted for a special pixel
    For each layer, compute the 4 discrete locations in the feature map
    closest to the sampled pixel!?
    based on http://blog.christianperone.com/2016/01/convolutional-hypercolumns-in-python/"""

    test_image = instance
    feature_maps = get_layer_outputs(model, layer_indexes, test_image, coordinates)
    #print(feature_maps)
    hypercolumns = []
    for idx, convmap in enumerate(feature_maps):
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(default_image_shape),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    print('shape of hypercolumns ', hypercolumns.shape)
    return hypercolumns


def extract_hypercolumn_matrix(model, layer_indexes, instance, pixel):
    """function returns a hypercoloumn matrix, consisting of hypercolumns for each sampled pixel"""
    test_image = instance
    hypercolumn_matrix = np.array(list)
    for p in pixel:
        print(p)
        hc = extract_hypercolumn(model, layer_indexes, test_image, p)
        hypercolumn_matrix.append(hc)
    return hypercolumn_matrix


"""number_features = 2 #needs to be set for the MLP layers!!!

# MLP layer where the hypercolumn vector goes into
MLP_layers = Sequential([
    Dense(1024, input_shape=(None)),
    Activation('relu'),
    Dense(1024),
    Activation('relu'),
    Dense(number_features),
    Activation('relu')
])"""


def main():
    csh = CityscapesHandler()

    # label handlers
    #print(csh.getClassIdFromName("car"))
    #print(csh.getClassNameFromId(12))
    #print(csh.getNumLabels())

    test = csh.getImageFromFilename("berlin_000000_000019_gtFine_color.png")

    # 1 hot transformations
    one_hot = csh.fromLabelIDsTo1hot([1, 2, 4, 5, 6, 6, 6, 7, 3, 2, 5, 5, 5, 0])
    back_translation = csh.from1hotToLabelIDs(one_hot)

    # read in 5 images of the different datasets
    train_x, train_y = csh.getTrainSet(5, asGreyScale=True)
    print("Trainx", train_x)
    print("Trainy", train_y)
    test_x, test_y = csh.getTestSet(5, asGreyScale=True)
    val_x, val_y = csh.getValSet(5, asGreyScale=True)

    # #get a numpy array of all read train_images
    # images = np.array(list(train_set.values()))

    # #print filenames of all loaded train images
    # print(train_set.keys())

    # display image
    csh.displayImage(train_y[0])
    #print(train_y[0])

    # generate random pixel samples for hypercolumn vectors
    samples = csh.samplePixels()
    #print(samples)
    #print(samples[0][0], samples[0][1])


    # compile model
    model1 = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")

    im_original = cv2.resize(train_y[0], (default_image_shape))
    test_image =  np.expand_dims(im_original, axis=0)
    print(test_image)

    # extract hypercolumns from different VGG layers
    layers_extract = [2, 4, 6, 8, 10, 12]
    hc = extract_hypercolumn_matrix(model1, layers_extract, test_image, samples)
    print(hc)
    """
    model = MLP_layers(hc)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    train_op = model.compile(optimizer=sgd, loss='categorical_crossentropy')

    iterations = 20
    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            print('EPOCH', i)
    
    ave = np.average(hc.transpose(1, 2, 0), axis=2)
    plt.imshow(ave)
    plt.show()"""


if __name__ == "__main__":
    main()


