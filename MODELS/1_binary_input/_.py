import cv2
import tensorflow as tf 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import TrainLogger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import regularizers

print("I HAVE STARTED")

# split function
def split_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# load data
try:
    train_data = np.load('../reviewed_train_dataset.npz')
    test_data = np.load('../reviewed_test_dataset.npz')
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Access the arrays in the .npz file
try:
    train_images = train_data['binary_images']
    train_labels = train_data['labels']
    test_images = test_data['binary_images']
    test_labels = test_data['labels']
    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
except KeyError as e:
    print(f"Key error: {e}")
    raise

y_train = train_labels
y_test = test_labels

# NORMALIZE THE DATA
X_train = tf.keras.utils.normalize(train_images, axis=1)
X_test = tf.keras.utils.normalize(test_images, axis=1)
print("Data normalized")

# RESHAPE THE IMAGE TO MAKE IT SUITABLE FOR APPLYING CONVOLUTION OPERATION
try:
    X_trainr = np.array(X_train).reshape(-1, 28, 28, 1)
    X_testr = np.array(X_test).reshape(-1, 28, 28, 1)
    print("Data reshaped")
except Exception as e:
    print(f"Error reshaping data: {e}")
    raise

print("I GOT HERE!")
