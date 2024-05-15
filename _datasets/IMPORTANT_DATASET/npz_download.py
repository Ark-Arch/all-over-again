import gzip
import numpy as np
import os

def load_emnist_data(dataset_path):
    images_file = os.path.join(dataset_path, 'emnist-bymerge-test-images-idx3-ubyte.gz')
    labels_file = os.path.join(dataset_path, 'emnist-bymerge-test-labels-idx1-ubyte.gz')

    with gzip.open(images_file, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(labels_file, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return images, labels

dataset_path = 'gzip'
images, labels = load_emnist_data(dataset_path)

# Save as .npz
np.savez('emnist_bymerge_test_dataset.npz', images=images, labels=labels)
