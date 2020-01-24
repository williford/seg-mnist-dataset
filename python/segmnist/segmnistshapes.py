import random
from textures import WhiteNoiseTexture
from segmnist import mnist_generator
import positioning as pos
# from . import loader
from texture_generator import random_color_texture
# from . import mnist_generator
import numpy as np
import math
from abc import ABCMeta

import sys

class RectangleShape(object):
    def __init__(self, center, length1, length2, orientation, texturegen):
        """
        Args:
            center: position of the center of the object (will be shifted by
                half a pixel, so that the center will be in center of the
                specified pixel.
            length1: length 1 of the rectangle
            length2: length 2 of the rectangle
            orientation: orientation of the rectangle (clockwise)
            texturegen: texture generator
        """
        self._center = center
        self._length1 = length1
        self._length2 = length2
        self._orientation = orientation
        self._texturegen = texturegen
        self._shape_mask = None

    def _calculate_shape_mask(self, imshape):
        yv, xv = np.meshgrid(
            np.arange(imshape[0]),
            np.arange(imshape[1]))
        dist_x = xv - (self._center[1] + 0.5)
        dist_y = yv - (self._center[0] + 0.5)

        # Rotation (around pos_yx)
        x_theta = (dist_x * np.cos(self._orientation()) - # Needs to be called because it is a function
                   dist_y * np.sin(self._orientation()))

        y_theta = (dist_x * np.sin(self._orientation()) +
                   dist_y * np.cos(self._orientation()))

        mask = np.minimum(
            self._length1 / 2.0 + 0.5 - abs(x_theta),
            self._length2 / 2.0 + 0.5 - abs(y_theta))
        mask = np.maximum(0.0, mask)
        mask = np.minimum(1.0, mask)
        return mask

    def draw_label_segimage(self, segmask, label):
        if self._shape_mask is None:
            self._shape_mask = self._calculate_shape_mask(segmask.shape)
        segmask[self._shape_mask > 0] = 255
        segmask[self._shape_mask >= 1] = label

    def draw_shape_image(self, image):
        if self._shape_mask is None:
            self._shape_mask = self._calculate_shape_mask(image.shape[1:])
        # texture = self._texturegen(
        #     image.shape,
        #     mean=np.random.randint(256, size=image.shape[0]),
        #     var=np.random.gamma(1, 25, size=image.shape[0]))
        texture = self._texturegen.generate(self._shape_mask)

        image[:] = (
            np.multiply(image, 1.0 - self._shape_mask) +
            np.multiply(texture, self._shape_mask))


class SquareShape(RectangleShape):
    def __init__(self, center, diameter, orientation, texturegen):
        super(SquareShape, self).__init__(
            center, diameter, diameter, orientation, texturegen)


class ShapeGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._frame = None
        self._max_ratio_outside = None  # express as ratio of diameter
        self._center_boundary = None
        self._drange = None

    def set_frame_boundary(self, frame_shape, max_shape_outside=0.0):
        assert len(frame_shape)==2
        self._frame = frame_shape
        self._max_ratio_outside = max_shape_outside
        assert max_shape_outside >=0 and max_shape_outside < 1.0

    def set_diameter_range(self, dmin, dmax):
        self._drange = (dmin, dmax)

    # @abstractmethod
    def generate_shape(self, positioning, texturegen):
        raise NotImplementedError

    # @abstractmethod
    def class_name(self):
        raise NotImplementedError


class RectangleGenerator(ShapeGenerator):
    def __init__(self, orientation_gen=lambda: random.uniform(0, math.pi/2)):
        super(RectangleGenerator, self).__init__()
        self._orientation_gen = orientation_gen

    def generate_shape(self, positioning, texturegen):
        ave_length = random.uniform(self._drange[0], self._drange[1])
        # length_diff is difference between lengths as a ratio
        # 0 = no difference, 1 = one side is twice as long as other
        length_diff = random.uniform(0.1, 1)
        if random.randint(0,1)==0:
            length1 = ave_length - ave_length * length_diff / 2.0
            length2 = ave_length + ave_length * length_diff / 2.0
        else:
            length1 = ave_length + ave_length * length_diff / 2.0
            length2 = ave_length - ave_length * length_diff / 2.0

        # center_x0 = self._frame.x0 + length1 - self._max_ratio_outside * length1
        # center_x1 = self._frame.x1 - length2 + self._max_ratio_outside * length2
        # center_x = random.uniform(center_x0, center_x1)
        # center_y0 = self._frame.y0 + length1 - self._max_ratio_outside * length1
        # center_y1 = self._frame.y1 - length2 + self._max_ratio_outside * length2
        # center_y = random.uniform(center_y0, center_y1)

        (center_x, center_y) = positioning.generate(
            self._frame, (length1, length2), center=True,
            max_ratio_outside=self._max_ratio_outside)
        orientation = self._orientation_gen

        return RectangleShape((center_y, center_x), length1, length2, orientation,
                           texturegen)

    def class_name(self):
        return 'Rectangle'


class SquareGenerator(ShapeGenerator):
    def __init__(self, orientation_gen=lambda: random.uniform(0, math.pi/2)):
        super(SquareGenerator, self).__init__()
        self._orientation_gen = orientation_gen
    # def __init__(self, texturegen):
    #     super(SquareGenerator, self).__init__(texturegen)

    def generate_shape(self, positioning, texturegen):
        # print(self._drange)
        diameter = random.uniform(self._drange[0], self._drange[1])

        (center_x, center_y) = positioning.generate(
            self._frame, (diameter, diameter), center=True,
            max_ratio_outside=self._max_ratio_outside)

        # center_x0 = self._frame.x0 + diameter - self._max_ratio_outside * diameter
        # center_x1 = self._frame.x1 - diameter + self._max_ratio_outside * diameter
        # center_x = random.uniform(center_x0, center_x1)

        # center_y0 = self._frame.y0 + diameter - self._max_ratio_outside * diameter
        # center_y1 = self._frame.y1 - diameter + self._max_ratio_outside * diameter
        # center_y = random.uniform(center_y0, center_y1)
        orientation = self._orientation_gen

        return SquareShape((center_y, center_x), diameter, orientation,
                           texturegen)

    def class_name(self):
        return 'Square'


class SegMNISTShapes(object):
    def __init__(self, mnist,
                 imshape,  # include nchannels: C x H x W
                 bg_pix_mul,
                 min_num_objects=1,
                 max_num_objects=None,
                 positioning='random',
                 shapes=[SquareGenerator(),
                         RectangleGenerator(),
                         ],
                 texturegen=None,
                 ):
        """
        bg_pix_mul: multiplier for the number of background pixels that are
            NOT masked out. If bg_pix_mul==1, then the number of background
            pixels will be set to be approximately the average number of pixels
            for each object / digit.
        """

        assert mnist is not None
        self._mnist_iter = mnist.iter()

        assert bg_pix_mul > 0.0, "bg_pix_mul must be more than 0."
        self._bg_pix_mul = bg_pix_mul

        self._digit_size = 28
        self._min_num_objects = min_num_objects
        self._max_num_objects = max_num_objects

        self._imshape = imshape

        self._generate_digit = mnist_generator.generate_digit

        if positioning == 'random':
            self._positioning = pos.RandomPositioning()
        elif positioning == 'grid':
            assert False, 'grid positioning disabled (at least for now)'
            self._positioning = pos.GridPositioning((2, 2))

        self._scale_range = (1.0, 1.0)

        if texturegen is None:
            texturegen = WhiteNoiseTexture(
                    mean_dist=lambda: np.random.randint(256),
                    var_dist=lambda: np.random.gamma(1, 25),
                    shape=imshape,
                    )
        self._texturegen = texturegen

        self._shapeGenerators = shapes
        for shape in self._shapeGenerators:
            shape.set_frame_boundary(imshape[1:3], 0.25)
            shape.set_diameter_range(
                self._digit_size * self._scale_range[0] * 0.25,
                self._digit_size * self._scale_range[1])
        self._class_names = [str(i) for i in range(10)]
        for shapeGen in self._shapeGenerators:
            self._class_names.append(shapeGen.class_name())

        self._classprob = None  # uniform by default

    def class_names(self):
        return self._class_names

    def set_class_freq(self, freq):
        self._classprob = np.array(freq, dtype=np.float)
        self._classprob[:] = self._classprob / np.sum(self._classprob[:])
        assert 10 + len(self._shapeGenerators) == self._classprob.size

    def set_min_digits(self, min_objects):
        self._min_num_objects = min_objects

    def set_max_digits(self, max_objects):
        self._max_num_objects = max_objects

    def set_nchannels(self, nchannels):
        self._imshape[0] = nchannels

    def set_scale_range(self, scale_range):
        self._scale_range = scale_range
        # for shape in self._shapeGenerators:
        #     shape.set_frame_boundary(self._imshape[1:3])
        #     shape.set_diameter_range(
        #         self._digit_size * self._scale_range[0],
        #         self._digit_size * self._scale_range[1])

        for shape in self._shapeGenerators:
            shape.set_frame_boundary(self._imshape[1:3], 0.25)
            shape.set_diameter_range(
                self._digit_size * self._scale_range[0] * 0.25,
                self._digit_size * self._scale_range[1])

    def set_bg_pix_mul(self, bg_pix_mul):
        """ Replaces seg_prob_mask_bg"""
        self._bg_pix_mul = bg_pix_mul

    """ Return single example with image, class labels, and segmentation labels.
    """
    def create_example(self):
        (img_data, cls_label, seg_label) = self.create_batch(1)
        img_data = img_data.reshape(img_data.shape[1:])
        seg_label = seg_label.reshape(seg_label.shape[2:])
        cls_label = cls_label.reshape(cls_label.shape[:2])
        return (img_data, cls_label, seg_label)

    """ Return batch with images, class labels, and segmentation labels.
        cls_label is a sparse vector with 1 set for every digit that
        appears in image.
    """
    def create_batch(self, batch_size):
        img_data = np.zeros((batch_size,
                             self._imshape[0],  # nchannels
                             self._imshape[1],  # 28 * self._gridH,
                             self._imshape[2]), dtype=np.uint8)
        seg_label = np.zeros((batch_size,
                              1,  # single channel
                              self._imshape[1],  # 28 * self._gridH,
                              self._imshape[2]), dtype=np.uint8)
        # print('PRINT ShapeGenerator: ', len(self._shapeGenerators), self._shapeGenerators)
        cls_label = np.zeros((batch_size,
                              10 + len(self._shapeGenerators), # Why is _shapeGenerators not 2???
                              # 12 + len(self._shapeGenerators), #Stijn edit
                              1, 1), dtype=np.uint8)

        for n in range(batch_size): # Loops trhough all batches
            nobj = random.randint(self._min_num_objects,
                                  self._max_num_objects)

            # positioning object needs to be reset each example
            #   (for finite sets)
            self._positioning.reset()
            # texture generator also needs to be reset
            self._texturegen.new_example(nobj+1)

            # create background texture
            seg_label[n] = np.zeros(shape=self._imshape[1:])
            img_data[n] = self._texturegen.generate()

            labels = set()
            # Add objects
            for iobj in range(nobj): # Loops through all objects
                # clsi = random.randint(0, 9 + len(self._shapeGenerators))
                clsi = np.random.choice(10 + len(self._shapeGenerators),
                                        p=self._classprob)
                if clsi > 9:
                    shape = (
                        self._shapeGenerators[clsi - 10].generate_shape(
                            self._positioning,
                            self._texturegen))

                    shape.draw_label_segimage(seg_label[n, 0], clsi + 1)
                    shape.draw_shape_image(img_data[n])
                    labels.add(clsi)
                else:
                    # clsi is ignored, class is determined by mnist iter
                    assert (
                        self._classprob is None or
                        np.all(np.isclose(
                            self._classprob[0],
                            self._classprob[1:10]))), (
                        'Digits are chosen with equal probability. '
                        'Having unequal digit class probabilities is '
                        'not supported.')
                    label = self._generate_digit(
                        img_data[n], seg_label[n, 0],
                        self._mnist_iter,
                        self._texturegen,
                        positioning=self._positioning)
                    labels.add(label)

            if self._bg_pix_mul > 0.0:  # should always be true
                npix_bg = np.sum(seg_label[n] == 0)
                npix_fg = np.sum(np.logical_and(
                    seg_label[n] > 0,
                    seg_label[n] < 255))

                # make npix_bg == npix_fg / nobj, when bg_pix_mul == 1
                if npix_bg == 0 or npix_bg == 0.: # patch
                    npix_bg = 1.
                prob_bg = min(
                        1.0, self._bg_pix_mul * (float(npix_fg) / nobj) / npix_bg)

                uniform = np.random.uniform(size=self._imshape[1:])
                # make sure there is atleast 1 bg pixel
                if np.sum(uniform[seg_label[n, 0] == 0]) == 0.: # patch
                    prob_pixel_bg = prob_bg
                else: 
                    prob_pixel_bg = max(
                    np.min(uniform[seg_label[n, 0] == 0]),
                    prob_bg)

                seg_label[n, 0][np.logical_and(
                    seg_label[n, 0] == 0,
                    uniform > prob_pixel_bg)] = 255

                npix_fg_2 = np.sum(np.logical_and(
                    seg_label[n] > 0,
                    seg_label[n] < 255))
                assert npix_fg == npix_fg_2

            assert seg_label[n].max() > 0 or nobj == 0, (
                ("The maximum value of the segmentation map is %f, "
                 "even though it has %d elements.") % (seg_label.max(), nobj))

            for lbl in labels:
                cls_label[n, lbl, 0, 0] = 1

            # Randomly pick a label
            # lbl = random.sample(labels, 1)[0]
            # cls_label[n, lbl, 0, 0] = 1

            # from PIL import Image
            # img = Image.fromarray(new_data, 'L')
            # img.show()
            # pdb.set_trace()
        return (img_data, cls_label, seg_label)
