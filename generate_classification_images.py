#!/usr/bin/env python

import os
import loader
import scipy.misc
import numpy as np
import errno
import pandas as pd

import pdb

from seg_mnist_2x2 import generate_segmnist_2x2_classification_images

import os
os.environ["MNIST_PATH"] = "/media/Data/Documents/Enkium/Projects/NIN/Data"

import sys
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python')

if __name__ == "__main__":
    generate_segmnist_2x2_classification_images()
