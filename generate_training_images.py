#!/usr/bin/env python

import os
import loader
import scipy.misc
import numpy as np
import errno
import pandas as pd

import pdb

from seg_mnist_2x2 import generate_segmnist_2x2_training_images


def mkdirs(path):
    """ Make directories, ignoring errors if directory already exists.
        Not needed for Python >=3.2.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass  # Ignore "File exists" errors
        else:
            raise


def generate_training_images():
    """ Generates training (and validation) images from the MNIST.  """
    if "MNIST_PATH" in os.environ:
        path = os.environ["MNIST_PATH"]
    else:
        path = os.path.expanduser("~/Data/mnist")

    training_all = loader.MNIST(path)
    training_all.load_standard('training')
    print("Number of training & validation examples: %d" % len(training_all))

    mnist = dict()
    mnist['trn'] = loader.MNIST(path, dataset_slice=(0, 50000))
    mnist['trn'].load_standard('training')
    print("Number of training examples: %d" % len(mnist['trn']))

    mnist['val'] = loader.MNIST(path, dataset_slice=(50000, 60000))
    mnist['val'].load_standard('training')
    print("Number of validation examples: %d" % len(mnist['val']))

    for subfolder in ['trn', 'val', 'tst']:
        mkdirs("seg-mnist-0/%s" % subfolder)

    # generate stimuli with 1 stimulus in a 3x3 grid
    miter = dict()  # mnist iterator
    miter['trn'] = mnist['trn'].iter(max_iter=float('inf'))
    miter['val'] = mnist['val'].iter(max_iter=float('inf'))

    columns = ['image', 'segmentation', 'x1', 'y1', 'x2', 'y2',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    columns_imglbl = ['image', 'label']

    for mode, within_grid_count in [('trn', 1000), ('val', 500)]:
        stimulus_number = 0
        imgseglbl = []  # image, segmented image, label information
        imglbl = []  # image, label information

        for pos in range(9):
            i = pos % 3
            j = pos // 3

            for within_grid in range(within_grid_count):
                label, data = miter[mode].next()

                w = data.shape[0]
                h = data.shape[1]
                offset_i = i * w
                offset_j = j * h
                new_data = np.zeros((3 * w, 3 * h), dtype=np.uint8)
                new_label = np.ones((3 * w, 3 * h), dtype=np.uint8) * 255

                new_data[offset_i:offset_i + w, offset_j:offset_j + h] = data
                new_label[offset_i:offset_i + w,
                          offset_j:offset_j + h][data > 0] = label

                fn_img = "seg-mnist-0/%s/mnistseg_%07d-image.png" % (
                    mode, stimulus_number)
                fn_seg = "seg-mnist-0/%s/mnistseg_%07d-segm.png" % (
                    mode, stimulus_number)
                scipy.misc.imsave(fn_img, new_data)
                scipy.misc.imsave(fn_seg, new_label)
                stimulus_number += 1

                one_hot_label = np.zeros(10, np.uint8)
                one_hot_label[label] = 1

                data = ["/" + fn_img, "/" + fn_seg, 0, 0, 3 * w, 3 * h]
                data.extend(one_hot_label.tolist())

                series = pd.Series(data, index=columns)
                imgseglbl.append(series)

                data = ["/" + fn_img, label]
                series = pd.Series(data, index=columns_imglbl)
                imglbl.append(series)
        
        imgseglbl_df = pd.DataFrame(imgseglbl, columns=columns)

        imgseglbl_df.to_csv("seg-mnist-0_%s.segmentation.txt" % mode, sep=' ', header=False, index=False)

        imglbl_df = pd.DataFrame(imglbl, columns=columns_imglbl)
        imglbl_df.to_csv("seg-mnist-0_%s.classification.txt" % mode, sep=' ', header=False, index=False)

    assert(len(mnist['trn']) + len(mnist['val']) == len(training_all))


if __name__ == "__main__":
    # generate_training_images()
    generate_segmnist_2x2_training_images()
