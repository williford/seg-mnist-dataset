#!/usr/bin/env python

import os
import loader
import scipy.misc
import numpy as np
import errno
import pandas as pd

import pdb

from seg_mnist_2x2 import generate_segmnist_2x2_training_images


if __name__ == "__main__":
    generate_segmnist_2x2_training_images()
