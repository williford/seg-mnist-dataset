import torch

# from segmnist import SegMNIST
from segmnist import SegMNISTShapes
from segmnist.loader.mnist import load_standard_MNIST
from segmnist.segmnistshapes import SquareGenerator
from segmnist.segmnistshapes import RectangleGenerator
# from segmnist.texture_generator import random_color_texture


class SegMNISTShapesPyTorch():
    """
    SegMNIST Shapes <https://github.com/williford/seg-mnist-dataset> Dataset.

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

    def __init__(self, batch_size, nclasses, im_shape, mnist_dataset,
                 bg_pix_mul=1.0, digit_positioning='random', max_digits=None,
                 min_digits=None, scale_range=(0.5, 1.5), GPU=False):
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.imshape = im_shape
        self.mnist_dataset_name = mnist_dataset
        self.bg_pix_mul = bg_pix_mul
        self.positioning = digit_positioning
        self.GPU = GPU

        self.mnist = load_standard_MNIST(
            self.mnist_dataset_name,
            shuffle=True
        )

        # self.mnist = SegMNIST.load_standard_MNIST(
        #     self.mnist_dataset_name,
        #     shuffle=True
        # )

        shapes = []
        # if self.nclasses >= 11:
        #     shapes.append(SquareGenerator(random_color_texture))
        # if self.nclasses >= 12:
        #     shapes.append(RectangleGenerator(random_color_texture))

        self.batch_loader = SegMNISTShapes(
            self.mnist,
            imshape=self.imshape,
            bg_pix_mul=self.bg_pix_mul,
            positioning=self.positioning,
            shapes=shapes
        )

        if max_digits:
            self.batch_loader.set_max_digits(max_digits)

        if min_digits:
            self.batch_loader.set_min_digits(min_digits)

        self.batch_loader.set_scale_range(scale_range)

        print_info("SegMNISTShapesLayerSync",
                   mnist_dataset,
                   batch_size,
                   im_shape)

    def get_batch(self, cuda_device=0):
        """
        Creates a batch. Returns torch tensors

        Args:
            cuda_device (int): CUDA device number
        """
        (img_data, cls_label, seg_label) = (
            self.batch_loader.create_batch(self.batch_size))

        # convert to tensor
        (img_data, cls_label, seg_label) = (
            torch.from_numpy(img_data).type(torch.FloatTensor),
            torch.from_numpy(cls_label).type(torch.FloatTensor),
            torch.from_numpy(seg_label).type(torch.FloatTensor))

        # flatten cls_label
        # print(img_data.shape, cls_label.shape, seg_label.shape, self.batch_size)
        # torch.Size([1024, 3, 56, 56]) torch.Size([1024, 10, 1, 1]) torch.Size([1024, 1, 56, 56])
        cls_label = cls_label.view(-1, self.nclasses)

        # transfer to GPU
        if self.GPU:
            (img_data, cls_label, seg_label) = (
                img_data.cuda(cuda_device),
                cls_label.cuda(cuda_device),
                seg_label.cuda(cuda_device))

        return (img_data, cls_label, seg_label)


def print_info(name, mnist_dataset, batch_size, im_shape):
    """
    Output some info regarding the class
    """
    print("{} initialized for dataset: {}, with bs: {}, im_shape: {}.".format(
        name, mnist_dataset, batch_size, im_shape))