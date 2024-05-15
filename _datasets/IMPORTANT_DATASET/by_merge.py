import numpy as np

# Load the .npz file
data = np.load('emnist_bymerge_test_dataset.npz')
data2 = np.load('emnist_bymerge_train_dataset.npz')

# Access the arrays in the .npz file
images = data['images']
labels = data['labels']

images2 = data2['images']
labels2 = data2['labels']

# Use the loaded data
print(images.shape[0] + images2.shape[0])
print(labels.shape[0] + labels2.shape[0])