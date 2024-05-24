import gzip
import numpy as np
import os
import cv2
from skimage.morphology import skeletonize

label_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 
    45: 'r', 46: 't'
}

# BINARIZE THE images
def binarize_image(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# GRADIENT MAGNITUDE
def compute_gradient_magnitude(image):
    _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Gx = cv2.Sobel(binarized, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(binarized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gradient_magnitude

# THIN THE IMAGE
def thinning(image):
    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Perform skeletonization (thinning)
    skeleton = skeletonize(binary_image / 255)
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton


# LOAD THE DATASET
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


print("ORIGINAL IMAGES ALREADY DOWNLOADED! waiting on others")
binarized_images = np.array([binarize_image(img) for img in images])
print("binaries now set")
gradiented_images = np.array([compute_gradient_magnitude(img) for img in images])
print("gradient now set")
thinned_images = np.array([thinning(img) for img in images])
print("thin now set. i.e. ALL SET!")

# Save as .npz
np.savez('emnist_bymerge_test_dataset.npz', images=images, binary_images=binarized_images, gradient_images=gradiented_images, thin_images=thinned_images,labels=labels)
