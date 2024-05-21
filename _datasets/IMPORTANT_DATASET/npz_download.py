import gzip
import numpy as np
import os

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
    images_file = os.path.join(dataset_path, 'emnist-bymerge-train-images-idx3-ubyte.gz')
    labels_file = os.path.join(dataset_path, 'emnist-bymerge-train-labels-idx1-ubyte.gz')

    with gzip.open(images_file, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(labels_file, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return images, labels

dataset_path = 'gzip'
images, labels = load_emnist_data(dataset_path)


print("ORIGINAL IMAGES ALREADY DOWNLOADED! waiting on others")
binarized_images = np.array([binarise_image(img) for img in images])
print("binaries now set")
gradiented_images = np.array([compute_gradient_magnitude(img) for img in images])
print("gradient now set")
thinned_images = np.array([thinning(img) for img in images])
print("thin now set. i.e. ALL SET!")

# Save as .npz
np.savez('emnist_bymerge_train_dataset.npz', images=images, binary_images=binarized_images, gradient_images=gradiented_images, thin_images=thinned_images,labels=labels)
