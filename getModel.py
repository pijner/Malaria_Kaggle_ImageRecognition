"""getModel.py: Script to handle operations on CNN model for this project"""

__author__ = "Prahar Ijner"
__email__ = "314jner@gmail.com"

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from sklearn.model_selection import StratifiedShuffleSplit

from getData import IMAGE_SIZE


class CNNModel:
    """
    Class to handle a Convolutional Neural Network (CNN) model
    """

    def __init__(self):
        self.model = self.buildModel()

    def buildModel(self):
        """
        Function to initialize and compile the CNN model
        :return: compiled CNN model
        """
        tf.compat.v1.reset_default_graph()
        model = Sequential()

        # First block
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(90, kernel_size=(7, 7), input_shape=IMAGE_SIZE + (3,), activation="relu",
                         data_format="channels_last", padding="same"))
        model.add(Conv2D(150, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(50, kernel_size=(5, 5), activation="relu", padding="same"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Second block
        model.add(Conv2D(25, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Third block
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(units=200, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=50, activation='relu'))

        # Output layer (prediction will be of shape (nImages, 2) in the form of normalized score for all classes)
        model.add(Dense(units=2, activation="softmax"))

        model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
        return model

    def modelTrain(self, trainData, trainLabels):
        """
        Function to train the compiled CNN model
        :param trainData: Images to train model on (shape: (nImages, image_height, image_width, 3))
        :param trainLabels: One hot encoded labels to for each image in trainData (shape: (nImages, 2))
        :return: None
        """
        # Keep 5% of training data for validation while training
        splits = StratifiedShuffleSplit(n_splits=2,
                                        train_size=0.95,
                                        test_size=0.05,
                                        random_state=0).split(trainData, trainLabels)
        trainIdx, valIdx = list(splits)

        self.model.fit(x=trainData[trainIdx[0], :, :, :],
                       y=trainLabels[trainIdx[0], :],
                       validation_data=(trainData[valIdx[0], :, :, :], trainLabels[valIdx[0], :]),
                       epochs=30,
                       verbose=1
                       )

    def modelPredict(self, testData):
        """
        Function to predict if a given image is parasitized or uninfected
        :param testData: Images to predict on (shape: (nImages, image_height, image_width, 3))
        :return: Normalized prediction score from softmax function (shape: (nImages, 2))
        """
        return self.model.predict(testData)

    def modelEvaluate(self, testData, testLabels):
        """
        Function to evaluate performance of the CNN model
        :param testData: Images to predict on (shape: (nImages, image_height, image_width, 3))
        :param testLabels: True one-hot encoded labels for each image in testData (shape: (nImages, 2))
        :return: None
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix, roc_auc_score

        predictions = self.modelPredict(testData)
        predictions_1D = np.argmax(predictions, axis=-1)
        test_1D = np.argmax(testLabels, axis=-1)
        error = np.sum(np.not_equal(predictions_1D, test_1D)) / np.shape(test_1D)[0]

        print("Error = ", error * 100, "%")
        print("Accuracy = ", (1 - error) * 100, "%")

        # Area under ROC curve
        print("AUC = ", roc_auc_score(test_1D, predictions_1D))
        # Confusion matrix [0: uninfected, 1: parasitized]
        print("Confusion matrix:")
        print(confusion_matrix(y_true=test_1D, y_pred=predictions_1D))
