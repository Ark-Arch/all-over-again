import numpy as np

# Load the dataset
data = np.load('emnist_bymerge_test_dataset.npz')

# Extract arrays
images = data['images']
binary_images = data['binary_images']
gradient_images = data['gradient_images']
thin_images = data['thin_images']
labels = data['labels']

# Label mapping
label_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 
    45: 'r', 46: 't'
}

# Desired characters
desired_characters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'E', 'N', 'G'}

# Get the indices of the desired characters
desired_indices = [index for index, char in label_mapping.items() if char in desired_characters]

# Filter the dataset
filtered_indices = np.isin(labels, desired_indices)

# Filtered arrays
filtered_images = images[filtered_indices]
filtered_binary_images = binary_images[filtered_indices]
filtered_gradient_images = gradient_images[filtered_indices]
filtered_thin_images = thin_images[filtered_indices]
filtered_labels = labels[filtered_indices]

# Save the filtered dataset
np.savez('reviewed_test_dataset.npz', images=filtered_images, binary_images=filtered_binary_images, gradient_images=filtered_gradient_images, thin_images=filtered_thin_images, labels=filtered_labels)

print('Filtered dataset saved to reviewed_test_dataset.npz')
