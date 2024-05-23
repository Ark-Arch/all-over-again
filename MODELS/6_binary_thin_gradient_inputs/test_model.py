import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import TrainLogger

from tensorflow.keras.models import load_model
model = load_model('model.keras')

# Split function
def split_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Load data
train_data = np.load('../emnist_bymerge_train_dataset.npz')
test_data = np.load('../emnist_bymerge_test_dataset.npz')

# INPUT PIPELINE FOR BINARY INPUT
train_bin_images = train_data['binary_images']
train_labels = train_data['labels']
test_bin_images = test_data['binary_images']
test_labels = test_data['labels']

y_train = train_labels
y_test = test_labels

# Normalize the data
X_train_bin = tf.keras.utils.normalize(train_bin_images, axis=1)
X_test_bin = tf.keras.utils.normalize(test_bin_images, axis=1)

# Reshape the image to make it suitable for applying convolution operation
X_train_bin = np.array(X_train_bin).reshape(-1, 28, 28, 1)
X_test_bin = np.array(X_test_bin).reshape(-1, 28, 28, 1)

# INPUT PIPELINE FOR THINNED IMAGES
train_thin_images = train_data['thin_images']
test_thin_images = test_data['thin_images']

# Normalize the data
X_train_thin = tf.keras.utils.normalize(train_thin_images, axis=1)
X_test_thin = tf.keras.utils.normalize(test_thin_images, axis=1)

# Reshape the image to make it suitable for applying convolution operation
X_train_thin = np.array(X_train_thin).reshape(-1, 28, 28, 1)
X_test_thin = np.array(X_test_thin).reshape(-1, 28, 28, 1)

# INPUT PIPELINE FOR GRADIENT IMAGES
train_grad_images = train_data['gradient_images']
test_grad_images = test_data['gradient_images']

# Normalize the data
X_train_grad = tf.keras.utils.normalize(train_grad_images, axis=1)
X_test_grad = tf.keras.utils.normalize(test_grad_images, axis=1)

# Reshape the image to make it suitable for applying convolution operation
X_train_grad = np.array(X_train_grad).reshape(-1, 28, 28, 1)
X_test_grad = np.array(X_test_grad).reshape(-1, 28, 28, 1)


X_testr = [X_test_bin, X_test_grad, X_test_thin]

# EVALUATING THE TESTING DATA
test_loss, test_acc = model.evaluate(X_testr, y_test)
print("Test loss on test samples", test_loss)
print("Validation Accuracy on test samples", test_acc)
