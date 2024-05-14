import tensorflow_datasets as tfds

# Load EMNIST dataset with ByMerge split
emnist = tfds.load('emnist', split='bymerge', as_supervised=True)

print(len(emnist))