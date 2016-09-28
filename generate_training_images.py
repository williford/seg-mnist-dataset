import os
import loader
import matplotlib.pyplot as plt
import scipy

import pdb


if __name__ == "__main__":
    if "MNIST_PATH" in os.environ:
        path = os.environ["MNIST_PATH"]
    else:
        path = os.path.expanduser("~/Data/mnist")

    training_all = loader.MNIST(path, slice_percent=(0, 100))
    training_all.load_standard('training')

    training = loader.MNIST(path, slice_percent=(0, 80))
    training.load_standard('training')

    validation = loader.MNIST(path, slice_percent=(80, 100))
    validation.load_standard('training')

   
    i = 0
    for label, data in training.random(max_iter=20):
        i += 1
        plt.imshow(data, cmap='gray', interpolation='nearest')
        plt.show()
        print(label)
