import torch
import random
import numpy as np

from segmnist import create_dataset_generator


class SegMNISTShapesPyTorch():
    """
    PyTorch wrapper for create_dataset_generator.

    See dataset_generator.py for arguments.

    Args:
        batch_size (int): Batch size
        nclasses (int): Number of classes (e.g. 12 for 10 digits plus
            squares and rectangles)
        im_shape (tuple): depth, height, width
        mnist_dataset (string): mnist-training or mnist-validation
        bg_pix_mul (float): multiplier for number of background pixels
            that are not masked
        digit_positioning (string): TODO
    """

    def __init__(self, **params):
                 # batch_size, nclasses, im_shape, mnist_dataset,
                 # bg_pix_mul=1.0, digit_positioning='random', max_digits=None,
                 # min_digits=None, scale_range=(0.5, 1.5), GPU=False):

        self.batch_loader = create_dataset_generator(**params)

        self.batch_size = params['batch_size']
        self.GPU = params['GPU']

        print_info("SegMNISTShapesPyTorch",
                   params['mnist_dataset'],
                   params['batch_size'],
                   params['im_shape'])

    def get_batch(self, segmentation=True, attend=False, cuda_device=0):
        """
        Creates a batch. Returns torch tensors:
        data            segMNIST image, shape (bs, depth, height, width)
        cls_label       Classes contained in image, shape (bs, nclasses)
        attend_label    Attended class, shape (bs, nclasses)
        seg_label       Segmentation result, shape (bs, depth, height)

        Args:
            segmentation (bool):    Return
                                    (data, cls_label, attend_label, seg_label)
                                    if True, otherwise return only
                                    (data, cls_label)
            cuda_device (int):      CUDA device number
        """
        (img_data, cls_label, seg_label_raw) = (
            self.batch_loader.create_batch(self.batch_size))

        # flatten not needed dimensions
        cls_label = cls_label.squeeze()
        seg_label_raw = seg_label_raw.squeeze()

        if segmentation:
            # set default unattended / background
            attend_label = np.zeros_like(cls_label)
            seg_label = np.zeros_like(seg_label_raw)

            for n in range(self.batch_size):
                # Create attend_label by randomly choosing one cls_label
                labels = list(np.flatnonzero(cls_label[n]))
                lbl = random.sample(labels, 1)[0]
                attend_label[n, lbl] = 1

                # retain masked out regions
                seg_label[n][seg_label_raw[n] == 255] = 255

                # set current label to foreground
                seg_label[n][seg_label_raw[n] == (lbl + 1)] = 1

            if attend:
                return_list = [img_data, cls_label, attend_label, seg_label]
            else:
                return_list = [img_data, cls_label, seg_label]

        else:
            return_list = [img_data, cls_label]

        # convert to tensor
        for i, arr in enumerate(return_list):
            return_list[i] = torch.from_numpy(arr).type(torch.FloatTensor)

        # transfer to GPU
        if self.GPU:
            for i, tensor in enumerate(return_list):
                return_list[i] = tensor.cuda(cuda_device)

        return return_list


def print_info(name, mnist_dataset, batch_size, im_shape):
    """
    Output some info regarding the class
    """
    print("{} initialized for dataset: {}, with bs: {}, im_shape: {}.".format(
        name, mnist_dataset, batch_size, im_shape))
