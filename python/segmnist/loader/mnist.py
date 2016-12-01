# Some code from https://gist.github.com/akesling/5358964

import os
import struct
import numpy as np

import random
import math


class MNIST(object):
    """ Class to read MNIST dataset and iterate examples in original order.
    """
    def __init__(self, root_dir="", dataset_slice=None, seed=2016):
        self.root_dir = root_dir
        self.lbl = None
        self.img = None
        assert dataset_slice is None or len(dataset_slice) == 2
        self.slice = dataset_slice
        self.seed_start = seed
        self.seed_current = seed

    def __len__(self):
        """ Return the number of images in dataset, or throw error if
            not initialized.
        """
        return len(self.lbl)

    def load_standard(self, name):
        """ Convenience function to load one of the standard MNIST datasets.
        """
        if name == 'training':
            return self.load('train-images-idx3-ubyte',
                             'train-labels-idx1-ubyte')
        elif name == 'testing':
            return self.load('t10k-images-idx3-ubyte',
                             't10k-labels-idx1-ubyte')
        raise RuntimeError('Unknown standard dataset "%s".' % name)

    def load(self, image_file, label_file):
        """ Load an image and label file pair in LeCunn's MNIST data format.
        """
        # set random seed for every new dataset
        self.seed_current = self.seed_start
        with open(os.path.join(self.root_dir, label_file), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        if len(lbl) == 0:
            raise RuntimeError("Dataset is empty.")

        with open(os.path.join(self.root_dir, image_file), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(
                len(lbl), rows, cols)

        if self.slice is None:
            begin = 0
            end = len(lbl)
        else:
            begin = self.slice[0]
            end = self.slice[1]
        self.lbl = lbl[begin:end]
        self.img = img[begin:end]

        return self.lbl, self.img


    def iter(self, max_iter=None):
        """ Iterate over the image and file pairs.
        
            By default, iterates over dataset once, but can be more (or infinite).
        """
        if self.lbl is None or self.img is None:
            raise RuntimeError("Dataset must be loaded before " +
                               "calling random().")

        if max_iter == None:
            max_iter = len(self.lbl)

        i=0
        while max_iter > 0:
            # set seed before each random range call
            yield (self.lbl[i % len(self.lbl)], self.img[i % len(self.lbl)])
            max_iter -= 1
            i+=1

    def is_shuffled(self):
        return False
