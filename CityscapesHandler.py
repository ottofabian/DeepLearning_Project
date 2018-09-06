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
    def __init__(self):
        self.trainId2label = labels.trainId2label
        self.id2label = labels.id2label
        
        self.fromLabelIdToTrainId = np.vectorize(self.__fromLabelIdToTrainId, otypes=[np.int])
        self.fromTrainIdToLabelId = np.vectorize(self.__fromTrainIdToLabelId, otypes=[np.int])

    def getNumLabels(self):
        return len(labels.labels) - 1

    def getNumTrainIDLabels(self):
        return len(labels.trainId2label.keys())

    def getClassNameFromId(self, class_id):
        return labels.id2label[class_id].name

    def getClassIdFromName(self, class_name):
        return labels.name2label[class_name].id

    def getImageFromFilename(self, filename):
        return csHelpers.getCsFileInfo(filename)

    def getDataset(self, setType, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False,
                   trainids=True):
        x = []
        y = []

        x_root = x_data_root + "/" + setType.lower()
        y_root = labels_data_root + "/" + setType.lower()

        if specificCity.lower() != "all":
            x_root += "/" + specificCity.lower()
            y_root += "/" + specificCity.lower()

        counter = 0
        finished = False
        for dirName, subdirList, fileList in os.walk(x_root):
            for fname in fileList:
                img = Image.open(dirName + "/" + fname)
                # print(dirName + "/" + fname)
                if asGreyScale:
                    img = img.convert("L")

                img = img.resize(shape)
                # x[csHelpers.getCoreImageFileName(fname)] = np.array(img)
                x.append(np.array(img))
                counter += 1

                if counter == maxNum:
                    finished = True
                    break
            if finished:
                break

        counter = 0
        finished = False
        for dirName, subdirList, fileList in os.walk(y_root):
            for fname in fileList:

                end = "_gtFine_labelTrainIds.png" if trainids else "_gtFine_labelIds.png"

                if fname.endswith(end):
                    img = Image.open(dirName + "/" + fname)
                    img = img.resize(shape)
                    # x[csHelpers.getCoreImageFileName(fname)] = np.array(img)
                    y.append(np.array(img))
                    counter += 1

                    if counter == maxNum:
                        finished = True
                        break
            if finished:
                break

        print(str(counter) + " images with shape " + str(shape) + " read for " + setType + "_set.")
        return np.array(x), np.array(y)

    def getTrainSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False):
        return self.getDataset("train", maxNum, specificCity, shape, asGreyScale)

    def getTestSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False):
        return self.getDataset("test", maxNum, specificCity, shape, asGreyScale)

    def getValSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False):
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

    # dummy implementation
    def samplePixels(self, numSamples=2000, imageShape=default_image_shape):
        result = []
        for k in range(0, numSamples):
            x = np.random.randint(0, high=imageShape[0] - 1)
            y = np.random.randint(0, high=imageShape[1] - 1)
            result.append(np.array([x, y]))

        return np.array(result)

    def displayImage(self, image):
        img = Image.fromarray(image)
        img.format = "PNG"
        img.show()
        
    
    def __fromLabelIdToTrainId(self, id):
        return self.id2label[id].trainId
        
        
    def __fromTrainIdToLabelId(self, trainId):
        return self.trainId2label[trainId].id
    

    def savePrediction(self, inputs, image_shape=(224, 224), oversample=True):

        # input_ = np.zeros((len(inputs),
        #                    image_shape[0],
        #                    image_shape[1],
        #                    inputs[0].shape[2]),
        #                   dtype=np.float32)
        #
        # for ix, in_ in enumerate(inputs):
        #     input_[ix] = resize_image(in_, self.image_dims)
        #
        # if oversample:
        #     # Generate center, corner, and mirrored crops.
        #     input_ = np.oversample(input_, self.crop_dims)
        # else:
        #     # Take center crop.
        #     center = np.array(self.image_dims) / 2.0
        #     crop = np.tile(center, (1, 2))[0] + np.concatenate([
        #         -self.crop_dims / 2.0,
        #         self.crop_dims / 2.0
        #     ])
        #     input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        #
        # # Classify
        # caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
        #                     dtype=np.float32)
        # for ix, in_ in enumerate(input_):
        #     caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        # out = self.forward_all(**{self.inputs[0]: caffe_in})
        # predictions = out[self.outputs[0]]
        #
        # # For oversampling, average predictions across crops.
        # if oversample:
        #     predictions = predictions.reshape((len(predictions) / 10, 10, -1))
        #     predictions = predictions.mean(1)
        #
        # return predictions
        #
        #
        # for img in predictions:
        #     img = Image.fromarray(img)
        #     # Needs to be save in ./results/Bremen/xxxx (or whatever)
        #     img.save('./results', format='PNG')
        pass


def main():
    csh = CityscapesHandler()
       
    test = np.array([1,2,3,4,5,6,7,8,9])
    
    print(csh.fromLabelIdToTrainId(test))
    print(csh.fromTrainIdToLabelId(test))

if __name__ == "__main__":
    main()
