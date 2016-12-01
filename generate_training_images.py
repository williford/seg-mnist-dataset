#!/usr/bin/env python

import os
import scipy.misc
import numpy as np
import errno
import pandas as pd

import pdb

from segmnist.static_segmnist2x2 import generate_segmnist_2x2_training_images


if __name__ == "__main__":
    generate_segmnist_2x2_training_images()
