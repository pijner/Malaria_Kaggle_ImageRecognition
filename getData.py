"""getData.py: Script to handle image read operations for Parasitized and Uninfected cell images"""

__author__ = "Prahar Ijner"
__email__ = "314jner@gmail.com"

import numpy as np
import glob
from PIL import Image

IMAGE_SIZE = (90, 90)


def resizeImage(imageFile, newSize=IMAGE_SIZE):
    """
    Function to resize image to fixed size
    :param imageFile: Image file obtained from Image.open() function in PIL.Image
    :param newSize: Tuple consisting of number of pixels for height and width to resize the image to (height, width)
    :return: resized image (same format as imageFile)
    """
    return imageFile.resize(newSize)


def getImages(path):
    """
    Function to read all images at a given path
    :param path: path to read images from
    :return: numpy array of images read from path (shape: (nImages, height, width, 3))
    """
    images = []
    for imgPath in glob.glob(path):
        imgFile = resizeImage(Image.open(imgPath))
        images.append(np.array(imgFile))

    return np.array(images)


def getParasitized(directory="."):
    """
    Function to read parasitized images and generate labels for them
    :param directory: path to directory containing cell_images directory (default location: current working directory)
    :return: list containing numpy array of parasitized images and numpy vector of ones (for labels)
    """
    print("Reading parasitized images...")
    pImages = getImages(directory + "/cell_images/Parasitized/*.png")
    print("Read complete")
    return [pImages, np.ones(np.shape(pImages)[0])]


def getUninfected(directory="."):
    """
    Function to read uninfected images and generate labels for them
    :param directory: path to directory containing cell_images directory (default location: current working directory)
    :return: list containing numpy array of parasitized images and numpy vector of zeroes (for labels)
    """
    print("Reading uninfected images...")
    uImages = getImages(directory + "/cell_images/Uninfected/*.png")
    print("Read complete")
    return [uImages, np.zeros(np.shape(uImages)[0])]
