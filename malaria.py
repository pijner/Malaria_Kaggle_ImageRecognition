"""malaria.py: Main script that uses getData and getModel to train and test the model"""

__author__ = "Prahar Ijner"
__email__ = "314jner@gmail.com"

import getData
import getModel

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical


def makeSplits(group1, group2, trainRatio=0.85):
    """
    Function to split parasited and uninfected data into train and test sets
    :param trainRatio: Percentage of dataset to use for training
    :param group1: list of images and labels for group1 in the form [images, labels]
    :param group2: list of images and labels for group2 in the form [images, labels]
    :return: list containing training dataset and testing dataset in the form
             [[trainingImages, trainingLabels], [testingImages, testingLabels]]

             trainingLabels and testingLabels returned are one-hot encoded
    """
    data = np.vstack((group1[0], group2[0]))
    labels = np.hstack((group1[1], group2[1]))

    labels = to_categorical(labels, num_classes=2)

    splits = StratifiedShuffleSplit(n_splits=2,
                                    train_size=trainRatio,
                                    test_size=1 - trainRatio,
                                    random_state=0).split(data, labels)
    trainIndex, testIndex = list(splits)
    trainData = data[trainIndex[0], :, :, :]
    testData = data[testIndex[0], :, :, :]

    trainLabels = labels[trainIndex[0], :]
    testLabels = labels[testIndex[0], :]

    return [[trainData, trainLabels], [testData, testLabels]]


def malaria():
    """
    Main function
    :return: None
    """
    # Get images from disk and create training and testing splits
    [[trX, trY], [tsX, tsY]] = makeSplits(getData.getParasitized(), getData.getUninfected())

    # Initialize and train model
    myModel = getModel.CNNModel()
    myModel.modelTrain(trainData=trX, trainLabels=trY)

    # Print model summary
    myModel.model.summary()

    # Assess model performance
    myModel.modelEvaluate(testData=tsX, testLabels=tsY)


malaria()
