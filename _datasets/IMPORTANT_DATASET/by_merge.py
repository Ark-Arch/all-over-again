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
print(f'the training images are {len(images2)}')
print(f'the testing images are {len(images)}')
print(f"total number of images {len(images) + len(images2)}")

combined_labels = np.concatenate((labels, labels2))
np.savez('the_labels.npz', labels=combined_labels)