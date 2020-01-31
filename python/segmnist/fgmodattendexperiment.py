import random
from segmnist.textures import FGModTexture
from segmnist.textures import WhiteNoiseTexture
from segmnist import mnist_generator
import numpy as np
from segmnist.segmnistshapes import SquareGenerator
from segmnist.positioning import RandomPositioning


class FGModAttendExperiment(object):
    def __init__(self, mnist,
                 imshape,  # include nchannels: C x H x W
                 bg_pix_mul,
                 texturegen,
                 texture_color_overlap,
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

        self._imshape = imshape

        self._generate_digit = mnist_generator.generate_digit

        self._scale_range = (1.0, 1.0)

        self._fgm_orientations = [
            np.stack([
                np.array([0]),
                np.array([0]),
            ]),
            np.stack([
                np.array([np.pi/2]),
                np.array([np.pi/2]),
            ]),
        ]
        self._texturegens = {
            'bg': FGModTexture(
                shape=self._imshape,
                independent_colors=True,
                texture_alpha=0.5,
                min_area_texture=100,
            ),
            'fg': FGModTexture(
                shape=self._imshape,
                independent_colors=True,
                texture_alpha=0.5,
                min_area_texture=100,
            ),
            'digit_dark':
                WhiteNoiseTexture(
                    mean_dist=lambda: 0,
                    var_dist=lambda: 0,
                    shape=self._imshape,
            ),
            'digit_white':
                WhiteNoiseTexture(
                    mean_dist=lambda: 255,
                    var_dist=lambda: 0,
                    shape=self._imshape,
            ),
        }
        self._digit_colors = (
            self._texturegens['digit_dark'],
            self._texturegens['digit_white'],
        )
        self._texture_color_overlap = texture_color_overlap
        self._fgmcolors = [
            lambda overlap: np.stack([
                np.full((3), 0),
                np.full((3), 255.0*(1.0 + overlap)/2),
            ]),
            lambda overlap: np.stack([
                np.full((3), 255.0*(1.0 - overlap)/2),
                np.full((3), 255.0),
            ]),
        ]

        sq = 20
        self._squareGenerator = SquareGenerator(orientation_gen=lambda: 0)
        for shape in [self._squareGenerator]:
            shape.set_frame_boundary(imshape[1:3], 0.25)
            shape.set_diameter_range(sq, sq)

        self._class_names = [str(i) for i in range(10)]
        for shapeGen in [self._squareGenerator]:
            self._class_names.append(shapeGen.class_name())
        self._class_names.append('Rectangle')

        self._classprob = None  # uniform by default
        ih = self._imshape[1]
        iw = self._imshape[2]
        self._positioning = [
            [  # Square is at top
                RandomPositioning(
                    range_pix_i=[ih/16+0.5,
                                 ih/16+0.5],
                    range_pix_j=[-sq/2, iw-sq/2]),
                RandomPositioning(
                    range_pix_i=[ih*0.5,
                                 ih*0.5],
                    range_pix_j=[iw*0,
                                 iw*0],
                                ),
                RandomPositioning(
                    range_pix_i=[ih*0.5,
                                 ih*0.5],
                    range_pix_j=[iw*0.5,
                                 iw*0.5],
                                ),
            ],
            [  # Square is at bottom
                RandomPositioning(
                    range_pix_i=[9.0*ih/16 + 0.5,
                                 9.0*ih/16 + 0.5],
                    range_pix_j=[-sq/2, iw-sq/2]),
                RandomPositioning(
                    range_pix_i=[ih*0,
                                 ih*0],
                    range_pix_j=[iw*0,
                                 iw*0],
                                ),
                RandomPositioning(
                    range_pix_i=[ih*0,
                                 ih*0],
                    range_pix_j=[iw*0.5,
                                 iw*0.5],
                                ),
            ],
            [  # Square is at left
                RandomPositioning(
                    range_pix_i=[-sq/2, iw-sq/2],
                    range_pix_j=[iw/16+0.5,
                                 iw/16+0.5]),
                RandomPositioning(
                    range_pix_i=[ih*0,
                                 ih*0],
                    range_pix_j=[iw*0.5,
                                 iw*0.5],
                                ),
                RandomPositioning(
                    range_pix_i=[ih*0.5,
                                 ih*0.5],
                    range_pix_j=[iw*0.5,
                                 iw*0.5],
                                ),
            ],
            [  # Square is at right
                RandomPositioning(
                    range_pix_i=[-sq/2, iw-sq/2],
                    range_pix_j=[9.0*iw/16+0.5,
                                 9.0*iw/16+0.5]),
                RandomPositioning(
                    range_pix_i=[ih*0,
                                 ih*0],
                    range_pix_j=[iw*0,
                                 iw*0],
                                ),
                RandomPositioning(
                    range_pix_i=[ih*0.5,
                                 ih*0.5],
                    range_pix_j=[iw*0,
                                 iw*0],
                                ),
            ],
        ]

    def set_min_digits(self, val):
        pass

    def set_max_digits(self, val):
        pass

    def class_names(self):
        return self._class_names

    def set_nchannels(self, nchannels):
        self._imshape[0] = nchannels

    def set_scale_range(self, scale_range):
        pass

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
        cls_label = np.zeros((
            batch_size,
            12,  # hard-coded - may need to be modifiable in the future
            1, 1), dtype=np.uint8)

        for n in range(batch_size):
            # show square iff nobj == 3
            nobj = random.sample([2, 3], 1)[0]

            # positioning object needs to be reset each example
            #   (for finite sets)
            for pos_group in self._positioning:
                for pos in pos_group:
                    pos.reset()

            overlap = np.random.uniform(
                np.min(self._texture_color_overlap),
                np.max(self._texture_color_overlap),
            )

            fgc, bgc = random.sample([0, 1], 2)
            # fg_cols, bg_cols = random.sample(self._fgmcolors, 2)
            fg_or, bg_or = random.sample(self._fgm_orientations, 2)
            self._texturegens['bg'].set_colors(self._fgmcolors[bgc](overlap))
            self._texturegens['fg'].set_colors(self._fgmcolors[fgc](overlap))
            self._texturegens['bg'].set_orientations(bg_or)
            self._texturegens['fg'].set_orientations(fg_or)

            # texture generators also needs to be reset
            for tgen in self._texturegens.values():
                tgen.new_example(1)

            # create background texture
            seg_label[n] = np.zeros(shape=self._imshape[1:])
            img_data[n] = self._texturegens['bg'].generate()
            # mean_bg = img_data[n].mean()
            # mean_digit = self._digit_colors[fgc]._mean_dist()

            pos_group = random.sample(self._positioning, 1)[0]
            labels = set()
            # Add objects
            # import pdb
            # pdb.set_trace()
            for iobj in range(nobj):
                if iobj == 0 and nobj == 3:  # square
                    shape = (
                        self._squareGenerator.generate_shape(
                            pos_group[iobj],
                            self._texturegens['fg']))
                    clsi = 10
                    shape.draw_label_segimage(seg_label[n, 0], clsi + 1)
                    shape.draw_shape_image(img_data[n])
                    labels.add(clsi)
                else:  # a digit
                    label = self._generate_digit(
                        img_data[n], seg_label[n, 0],
                        self._mnist_iter,
                        self._digit_colors[fgc],
                        #self._texturegens['digit'],
                        positioning=pos_group[iobj + 3 - nobj])
                    labels.add(label)

            if self._bg_pix_mul > 0.0:  # should always be true
                npix_bg = np.sum(seg_label[n] == 0)
                npix_fg = np.sum(np.logical_and(
                    seg_label[n] > 0,
                    seg_label[n] < 255))

                # make npix_bg == npix_fg / nobj, when bg_pix_mul == 1
                prob_bg = min(
                    1.0, self._bg_pix_mul * (float(npix_fg) / nobj) / npix_bg)

                uniform = np.random.uniform(size=self._imshape[1:])
                # make sure there is atleast 1 bg pixel
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

    def set_class_freq(self, freq):
        pass
