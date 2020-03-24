from sys import argv
from pathlib import Path

# Keras/machine learning imports
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import losses
from keras import metrics
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback
#from keras_tqdm import TQDMCallback
from keras.models import load_model

from configparser import ConfigParser, ExtendedInterpolation

# Read config file for paths
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


# Load model
print("Loading Model")
p = config["Paths"]["models"] + "/" + argv[1]
model = load_model(p)

# Predict
print("Predicting")
test_data = np.load(f"{config['Paths']['ising data']}/test_data.npz")["data"]
test_data = test_data.reshape(len(test_data), 128, 128, 1)
predictions = model.predict(test_data)

# Save predictions
name = config["Paths"]["predictions"] + "/" + argv[1].strip(".h5") + "_predictions.npy"
np.save(name, predictions)