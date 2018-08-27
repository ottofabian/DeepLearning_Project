from data.cityscapesscripts.helpers import csHelpers
from data.cityscapesscripts.helpers import labels

from sklearn.preprocessing import LabelEncoder

import PIL
from PIL import Image
import scipy as sp
import os
import numpy as np


x_data_root = "./data/leftImg8bit"
labels_data_root = "./data/gtFine"
default_image_shape = (224, 224)
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
    test_x, test_y = csh.getTestSet(5, asGreyScale=True)
    val_x, val_y = csh.getValSet(5, asGreyScale=True)

    # #get a numpy array of all read train_images
    # images = np.array(list(train_set.values()))

    # #print filenames of all loaded train images
    # print(train_set.keys())

    # display image
    #csh.displayImage(train_y[0])
    #print(train_y[0])

    # generate random pixel samples for hypercolumn vectors
    samples = csh.samplePixels()
    #print(samples)
    #print(samples[0][0], samples[0][1])


if __name__ == "__main__":
    main()


