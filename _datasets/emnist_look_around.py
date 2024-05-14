from emnist import extract_training_samples, extract_test_samples

# Load the training dataset
X_train, y_train = extract_training_samples('bymerge')

"""
# Load the test dataset
X_test, y_train = extract_test_samples('bymerge')

# Print the shapes of the datasets
print("Training dataset shape:", X_train.shape, y_train.shape)
print("Test dataset shape:", X_test.shape, y_test.shape)
"""
