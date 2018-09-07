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
prediction_save_path = "./data/results"
default_image_shape = (224, 224)
original_image_shape = (2048, 1024)
useTrainingLabels = False


class CityscapesHandler(object):
    def __init__(self):
        self.trainId2label = labels.trainId2label
        self.id2label = labels.id2label

        self.fromLabelIdToTrainId = np.vectorize(self.__fromLabelIdToTrainId, otypes=[np.int])
        self.fromTrainIdToLabelId = np.vectorize(self.__fromTrainIdToLabelId, otypes=[np.int])
        self.getColorFromLabelId = np.vectorize(self.__getColorFromLabelId, otypes=[np.int])

    def getNumLabels(self):
        return len(labels.labels) - 1

    def getNumTrainIDLabels(self):
        return len(labels.trainId2label.keys()) - 1

    def getClassNameFromId(self, class_id):
        return labels.id2label[class_id].name

    def getClassIdFromName(self, class_name):
        return labels.name2label[class_name].id

    def getImageFromFilename(self, filename):
        return csHelpers.getCsFileInfo(filename)

    def getDataset(self, setType, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False,
                   trainids=True, withFilenames=False):
        x = []
        y = []

        listFilenames = []
        listFilenamesLabels = []

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
                listFilenames.append(fname)
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
                    listFilenamesLabels.append(fname)
                    counter += 1

                    if counter == maxNum:
                        finished = True
                        break
            if finished:
                break

        print(str(counter) + " images with shape " + str(shape) + " read for " + setType + "_set.")

        if (withFilenames):
            return np.array(x), np.array(y), listFilenames, listFilenamesLabels
        else:
            return np.array(x), np.array(y)

    def getTrainSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False,
                    withFilenames=False):
        return self.getDataset("train", maxNum, specificCity, shape, asGreyScale, withFilenames=withFilenames)

    def getTestSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False,
                   withFilenames=False):
        return self.getDataset("test", maxNum, specificCity, shape, asGreyScale, withFilenames=withFilenames)

    def getValSet(self, maxNum=-1, specificCity="all", shape=default_image_shape, asGreyScale=False,
                  withFilenames=False):
        return self.getDataset("val", maxNum, specificCity, shape, asGreyScale, withFilenames=withFilenames)

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
        try:
            return self.trainId2label[trainId].id
        except KeyError:
            return trainId

    def __getColorFromLabelId(self, id):
        return self.id2label[id].color

    def fromInputFilenamesToPredictionFilenames(self, filenames):
        predictionFilenames = []

        for name in filenames:
            name = name.split("_")
            predictionFilenames.append(name[0] + "_" + name[1] + "_" + name[2] + "_" + "prediction.png")

        for e in predictionFilenames:
            print(e)

        return predictionFilenames

    def savePrediction(self, image, filename, image_shape=(224, 224)):

        image = self.fromTrainIdToLabelId(image)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image)

        image = image.resize(image_shape)

        image.save(prediction_save_path + "/" + filename)


def main():
    csh = CityscapesHandler()

    test = np.array([255])

    a = csh.fromTrainIdToLabelId(test)

    r, g, b = csh.getColorFromLabelId(a[0])
    print(r, g, b)


if __name__ == "__main__":
    main()
