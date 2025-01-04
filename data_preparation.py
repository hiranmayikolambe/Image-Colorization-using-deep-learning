# data_preparation.py
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

def preprocess_images():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Convert images to LAB color space
    x_train_lab = cv2.cvtColor(x_train, cv2.COLOR_RGB2LAB)
    x_test_lab = cv2.cvtColor(x_test, cv2.COLOR_RGB2LAB)

    # Split into L and AB channels
    L_train = x_train_lab[:, :, :, 0] / 255.0
    AB_train = x_train_lab[:, :, :, 1:] / 128.0  # Normalize AB channels

    L_test = x_test_lab[:, :, :, 0] / 255.0
    AB_test = x_test_lab[:, :, :, 1:] / 128.0

    # Reshape for model input
    L_train = L_train.reshape(-1, 32, 32, 1)
    AB_train = AB_train.reshape(-1, 32, 32, 2)
    L_test = L_test.reshape(-1, 32, 32, 1)
    AB_test = AB_test.reshape(-1, 32, 32, 2)

    return (L_train, AB_train), (L_test, AB_test)

if __name__ == "__main__":
    preprocess_images()