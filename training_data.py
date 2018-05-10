import cv2
import glob
import numpy as np
import os
import pickle
from skimage.feature import hog

TRAINING_DATA_FILE = "training_data.pkl"

def load_training_data():
    vehiclesTrainingData = load_vehicles_training_data()
    nonVehiclesTrainingData = load_non_vehicles_training_data()
    return concatenate_training_data([
        vehiclesTrainingData,
        nonVehiclesTrainingData
    ])

def load_vehicles_training_data():
    print("Loading vehicles training data ...")
    return load_training_data_from_directory(
        "training_images/vehicles", 1)

def load_non_vehicles_training_data():
    print("Loading non-vehicles training data ...")
    return load_training_data_from_directory(
        "training_images/non-vehicles", 0)

def load_training_data_from_directory(directory, label):
    imagePaths = glob.glob(os.path.join(directory, "**/*.png"))
    features = list()
    labels = list()
    for path in imagePaths:
        image = cv2.imread(path)
        imageFeatures = obtain_features(image)
        features.append(imageFeatures)
        labels.append(label)
    return features, labels

def obtain_features(image):
    return np.concatenate((
        hog(image[:,:,0], orientations=9, pixels_per_cell=(8, 8)),
        hog(image[:,:,1], orientations=9, pixels_per_cell=(8, 8)),
        hog(image[:,:,2], orientations=9, pixels_per_cell=(8, 8)),
    ))

def concatenate_training_data(trainingDataList):
    features = list()
    labels = list()
    for entry in trainingDataList:
        features += entry[0]
        labels += entry[1]
    return np.float32(features), np.float32(labels)

def save(data):
    print("Saving training data ...")
    with open(TRAINING_DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def load():
    print("Loading training data ...")
    with open(TRAINING_DATA_FILE, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":

    trainingData = load_training_data()
    save(trainingData)
