#!/usr/bin/env python

import os
import loader
import scipy.misc
import numpy as np
import errno
import pandas as pd
import itertools

import pdb

import h5py
from . import generator as gen
from generator import generate_textured_grid
from generator import generate_textured_image


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


def generate_segmnist_2x2_classification_images():

    flist = generate_segmnist_2x2_training_images_bgmask(mask_bg=False)

    for f1 in flist:
        task = f1[0]
        mode = f1[1]
        if task != 'classification':
            continue

        with open("%s-segmnist-2x2-any.%s.txt" % (task, mode), 'w') as f:
            for line in f1[2]:
                f.write(line + '\n')


def generate_segmnist_2x2_training_images():

    flist = generate_segmnist_2x2_training_images_bgmask(mask_bg=False)
    flist_bgmask = generate_segmnist_2x2_training_images_bgmask(mask_bg=True)

    for (f1, f2) in zip(flist, flist_bgmask):
        task = f1[0]
        mode = f1[1]
        with open("%s-segmnist-2x2-any.%s.txt" % (task, mode), 'w') as f:
            for line in f1[2]:
                f.write(line + '\n')
            if task != 'classification' or flist_bgmask is None:
                for line in f2[2]:
                    f.write(line + '\n')


def generate_segmnist_2x2_training_images_bgmask(mask_bg):
    # generate_segmnist_2x2_all_training_images(mask_bg=True)
    # generate_segmnist_2x2_1_training_images()
    filelists = list()
    for ncells in range(1, 5):
        filelists.append(
            generate_segmnist_2x2_all_training_images(cells_with_num=ncells,
                                                      mask_bg=mask_bg))

    if mask_bg:
        mask_bg_str = 'with_bgmask'
    else:
        mask_bg_str = 'without_bgmask'

    filelists2 = [
        ('classification', 'trn',
         [fn for file_quad in filelists for fn in file_quad['trn_cls']],
         mask_bg_str),
        ('segmentation', 'trn',
         [fn for file_quad in filelists for fn in file_quad['trn_seg']],
         mask_bg_str),
        ('classification', 'val',
         [fn for file_quad in filelists for fn in file_quad['val_cls']],
         mask_bg_str),
        ('segmentation', 'val',
         [fn for file_quad in filelists for fn in file_quad['val_seg']],
         mask_bg_str)]

    for (task, mode, filenames, mask_bg_str) in filelists2:
        fn = "%s-segmnist-2x2-any_%s.%s.txt" % (task, mask_bg_str, mode)
        print fn
        with open(fn, 'w') as f:
            for line in filenames:
                f.write(line + '\n')

    return filelists2


def generate_segmnist_2x2_all_training_images(cells_with_num, mask_bg=False):
    if not mask_bg:
        output_dir = 'seg-mnist-2x2-%d' % cells_with_num
    else:
        output_dir = 'seg-mnist-2x2-%d-bgmask' % cells_with_num

    for subfolder in ['trn', 'val', 'tst']:
        mkdirs("%s/%s" % (output_dir, subfolder))

    if "MNIST_PATH" in os.environ:
        path = os.environ["MNIST_PATH"]
    else:
        path = os.path.expanduser("~/Data/mnist")

    mnist_trn = loader.MNIST(path, dataset_slice=(0, 5000))
    mnist_trn.load_standard('training')

    mnist_val = loader.MNIST(path, dataset_slice=(5000, 6000))
    mnist_val.load_standard('training')

    (hclsfiles_trn, hsegfiles_trn) = (
        generate_segmnist_2x2_x_images(dataset=mnist_trn,
                                       output_dir=output_dir,
                                       mode='trn',
                                       #num_examples=50 * 1000,
                                       num_examples=50,
                                       cells_with_num=cells_with_num,
                                       mask_bg=mask_bg,
                                       seed_offset=0))

    (hclsfiles_val, hsegfiles_val) = (
        generate_segmnist_2x2_x_images(dataset=mnist_val,
                                       output_dir=output_dir,
                                       mode='val',
                                       # num_examples=10 * 1000,
                                       num_examples=10,
                                       cells_with_num=cells_with_num,
                                       mask_bg=mask_bg,
                                       seed_offset=50 * 1000))
    return dict(
        trn_cls=hclsfiles_trn,
        trn_seg=hsegfiles_trn,
        val_cls=hclsfiles_val,
        val_seg=hsegfiles_val,
    )


def generate_segmnist_2x2_x_images(dataset, output_dir, mode, num_examples,
                                   cells_with_num, mask_bg=False,
                                   seed_offset=0):
    """ Generates 2x2 grid where some cells contain an MNIST number.

        dataset - source MNIST dataset loader
        cells_with_num - the number of cells with a number
    """
    assert cells_with_num > 0 and cells_with_num <= 4
    mnist_iter = dataset.iter(max_iter=float('inf'))

    columns = ['image', 'segmentation', 'x1', 'y1', 'x2', 'y2',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    columns_imglbl = ['image', 'label']
    imgseglbl = []  # image, segmented image, label information
    imglbl = []  # image, segmented image, label information

    # create iterator that cycles through all permutations of arrangements
    grid_arrangements = itertools.cycle(
        itertools.permutations(
            [True] * cells_with_num +
            [False] * (4 - cells_with_num)))

    hsegfiles = []
    hclsfiles = []

    for stimulus_number in range(num_examples):
        np.random.seed(stimulus_number + seed_offset)
        grid = np.array(grid_arrangements.next(), dtype=np.bool).reshape(2, 2)
        # (new_data, new_seg_label, labels) = (
        #     generate_textured_grid(mnist_iter, grid, bgmask=mask_bg))
        (new_data, new_seg_label, labels) = (
            generate_textured_image(mnist_iter, grid, bgmask=mask_bg))

        (group, group_remainder) = divmod(stimulus_number, 1000)
        if group_remainder == 0:
            mkdirs("%s/%s/%d" % (output_dir, mode, group))
            mkdirs("%s/hdf5/%s/%d" % (output_dir, mode, group))

        fn_img = "%s/%s/%d/%s_%s_%07d-image.png" % (
            output_dir,
            mode,
            group,
            output_dir,
            mode,
            stimulus_number)
        fn_seg = "%s/%s/%d/%s_%s_%07d-segm.png" % (
            output_dir,
            mode,
            group,
            output_dir,
            mode,
            stimulus_number)
        fn_hdf5_cls = "%s/hdf5/%s/%d/%s_%s_%07d.cls.hdf5" % (
            output_dir,
            mode,
            group,
            output_dir,
            mode,
            stimulus_number)
        hclsfiles.append(fn_hdf5_cls)

        scipy.misc.imsave(fn_img, new_data)
        scipy.misc.imsave(fn_seg, new_seg_label)
        stimulus_number += 1

        with h5py.File(fn_hdf5_cls, 'w') as f:
            if new_data.ndim == 3:
                new_shape = (1,) + new_data.shape
            else:
                new_shape = (1,1) + new_data.shape

            f['data'] = new_data.reshape(new_shape)

            # sparse but can contain multiple labels
            sparse_labels = np.zeros(10, np.uint8)
            for lbl in set(labels):
                sparse_labels[lbl] = 1

            f['cls-label'] = sparse_labels.reshape((1, 10, 1, 1))

        example_inf = [fn_img, fn_seg, 0, 0, new_data.shape[0],
                       new_data.shape[1]]

        for (lbl_index, lbl) in enumerate(set(labels)):
            one_hot_label = np.zeros(10, np.uint8)
            one_hot_label[lbl] = 1

            example_inf1 = list(example_inf)
            example_inf1.extend(one_hot_label)

            series = pd.Series(example_inf1, index=columns)
            imgseglbl.append(series)

            fn_hdf5_seg = "%s/hdf5/%s/%d/%s_%s_%07d-%d.seg.hdf5" % (
                output_dir,
                mode,
                group,
                output_dir,
                mode,
                stimulus_number,
                lbl_index)

            with h5py.File(fn_hdf5_seg, 'w') as f:
                if new_data.ndim == 3:
                    new_shape = (1,) + new_data.shape
                else:
                    new_shape = (1,1) + new_data.shape

                f['data'] = new_data.reshape(new_shape)
                f['cls-label'] = one_hot_label.reshape((1, 10, 1, 1))

                # set default values to be background
                # (used for the digits with other labels)
                seg_label_fg = np.zeros((1, 1, new_seg_label.shape[0],
                                         new_seg_label.shape[1]))
                seg_label_bg = np.ones((1, 1, new_seg_label.shape[0],
                                        new_seg_label.shape[1]))

                # retain masked out regions
                # (if mask_bg, this includes the original background)
                seg_label_fg[0, 0][new_seg_label == 255] = 255
                seg_label_bg[0, 0][new_seg_label == 255] = 255

                # set current label to foreground
                seg_label_fg[0, 0][new_seg_label == lbl + 1] = 1
                seg_label_bg[0, 0][new_seg_label == lbl + 1] = 0

                # set other labels to be background
                # f['seg-label'] = np.concatenate((seg_label_bg, seg_label_fg),
                #   axis=1)
                f['seg-label'] = seg_label_fg

            hsegfiles.append(fn_hdf5_seg)

        if cells_with_num == 1:
            label = list(labels)[0]
            data = ["/" + fn_img, label]
            series = pd.Series(data, index=columns_imglbl)
            imglbl.append(series)

    imgseglbl_df = pd.DataFrame(imgseglbl, columns=columns)

    imgseglbl_df.to_csv("%s_%s.segmentation.txt" % (output_dir, mode),
                        sep=' ', header=False, index=False)

    with open("%s-%s-classification-hdf5.txt" % (output_dir, mode), 'w') as f:
        for fn in hclsfiles:
            f.write(fn + '\n')

    with open("%s-%s-segmentation-hdf5.txt" % (output_dir, mode), 'w') as f:
        for fn in hsegfiles:
            f.write(fn + '\n')

    if len(imglbl) > 0:
        imglbl_df = pd.DataFrame(imglbl, columns=columns_imglbl)
        imglbl_df.to_csv("%s_%s.classification.txt" % (output_dir, mode),
                         sep=' ', header=False, index=False)

    return (hclsfiles, hsegfiles)
