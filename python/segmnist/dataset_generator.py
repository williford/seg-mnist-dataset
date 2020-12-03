import numpy as np
import random

from segmnist import SegMNISTShapes
from segmnist import FGModAttendExperiment
from segmnist import StimSetDispatcher

from segmnist.loader import load_standard_MNIST
from segmnist.segmnistshapes import SquareGenerator
from segmnist.segmnistshapes import RectangleGenerator

from segmnist.textures import TextureDispatcher
from segmnist.textures import WhiteNoiseTexture
from segmnist.textures import SinusoidalGratings
from segmnist.textures import FGModTexture
from segmnist.textures import IntermixTexture

def create_dataset_generator(seed=None, **params):
    """ Overall SegMNIST generator.

    Args:
        seed - random seed to allow reproducibility (set to None for training)
        params should be a dict of the following:
        mnist_dataset - ex: mnist-training, determines data partition of MNIST
        digit_positioning - use 'random', other values may be deprecated?
        scale_range: ex: (0.8, 1.2)
        im_shape: ex: (3, 56, 56), the shape of the canvas / image
        bg_pix_mul: ex: 3, can be use to help deal with class imbalance of
            background vs. attended vs unattended foreground objects.
            More intelligent loss functions, like class-balanced focal loss?
        batch_size: not actually used by this method, batch_size is passed to
            the returned generator.
        min_digits: minimum number of **objects**
        max_digits: maximum number of **objects**
        nclasses: use 12, for 10 digits and 2 shapes
        p_fgmodatt_set: e.g. 0.75               JW-TODO: explain following
        fgmodatt_color_overlap: ex. (1,1)
        pwhitenoise: 0
        pgratings: 0
        pfgmod: 0.25
        fgmod_indepcols: 0.2
        fgmod_texalpha: (0.75,1.0)
        fgmod_min_area: 0
        pintermix: 0.05
        classfreq: ex (1,1,1,1,1, 1,1,1,1,1, 5.0,5.0). Can control the
            frequency of each class. Example makes a digit just as like as a
            square or rectangle (a square appearing is 5 times more likely
            than any given digit).
    """
    np.random.seed(seed)
    random.seed(seed)

    # Check the parameters for validity.
    check_params(params)

    # multiplier for number of background pixels that are not masked
    # replaces prob_mask_bg
    if 'bg_pix_mul' in params.keys():
        bg_pix_mul = params['bg_pix_mul']
    else:
        bg_pix_mul = 1.0

    # -------------------------------------------------------------------
    # Read in parameters for the general stimulus set general_stimset
    # The probability of a stimulus being from this stimulus set is
    # (1 - p_fgmodatt_set).
    # The general_stimset encapsulates a lot of random stimulus
    # posibilities.
    # -------------------------------------------------------------------

    imshape = params['im_shape']

    if 'digit_positioning' in params.keys():
        positioning = params['digit_positioning']
    else:
        positioning = 'random'

    if 'nchannels' in params.keys():
        raise NotImplementedError(
            "Using nchannels is currently not "
            "implemented. Use im_shape instead.")

        # assumes imshape is a tuple
        imshape = (params['nchannels'],) + imshape

    # Create a batch loader to load the images.
    mnist = load_standard_MNIST(
        params['mnist_dataset'],
        shuffle=True,
        seed=seed,
    )

    shapes = []
    if params['nclasses'] >= 11:
        shapes.append(SquareGenerator())
    if params['nclasses'] >= 12:
        shapes.append(RectangleGenerator())

    texturegen = TextureDispatcher()
    gratings = None
    if 'pgratings' in params.keys() and params['pgratings'] > 0:
        gratings = SinusoidalGratings(shape=imshape)
        texturegen.add_texturegen(
            params['pgratings'],
            gratings)

    fgmod = None
    if 'pfgmod' in params.keys() and params['pfgmod'] > 0:
        if 'fgmod_min_area' in params.keys():
            min_area = params['fgmod_min_area']
        else:
            min_area = imshape[0] * imshape[1] / 16

        if 'fgmod_indepcols' in params.keys():
            indepcols = params['fgmod_indepcols']
        else:
            # 0 means all textures have same color within example
            indepcols = 0.0

        # if 'fgmod_texalpha' in params.keys():
        texalpha = params['fgmod_texalpha']
        # else:
        #     texalpha = 1.0  # 0=no textures, 1="full" texture

        fgmod = FGModTexture(
            shape=imshape,
            independent_colors=indepcols,
            texture_alpha=texalpha,
            min_area_texture=min_area,
        )
        texturegen.add_texturegen(
            params['pfgmod'],
            fgmod)

    defaultTexture = WhiteNoiseTexture(
        mean_dist=lambda: np.random.randint(256),
        var_dist=lambda: np.random.gamma(1, 25),
        shape=imshape,
    )
    if 'pwhitenoise' in params.keys() and params['pwhitenoise'] > 0:
        texturegen.add_texturegen(
            params['pwhitenoise'],
            defaultTexture)

    if 'pintermix' in params.keys() and params['pintermix'] > 0:
        randomtex = IntermixTexture()
        if gratings is not None:
            randomtex.add_texturegen(
                params['pgratings'],
                gratings)
        if fgmod is not None:
            randomtex.add_texturegen(
                params['pfgmod'],
                fgmod)
        if 'pwhitenoise' in params.keys() and params['pwhitenoise'] > 0:
            randomtex.add_texturegen(
                params['pwhitenoise'],
                defaultTexture)

        texturegen.add_texturegen(
            params['pintermix'],
            randomtex)

    general_stimset = SegMNISTShapes(
        mnist,
        imshape=imshape,
        bg_pix_mul=bg_pix_mul,
        positioning=positioning,
        shapes=shapes,
        texturegen=texturegen,
    )
    # -------------------------------------------------------------------
    # Create the FGMod Attention Stimulus set. This is for the experiment
    # that corresponds to the Poort et al 2012 experiment.
    # The probability of a stimulus being drawn from this set is
    # p_fgmodatt_set.
    # -------------------------------------------------------------------

    if 'p_fgmodatt_set' in params.keys() and params['p_fgmodatt_set'] > 0:
        texture_color_overlap = (0, 0)
        if 'fgmodatt_color_overlap' in params.keys():
            texture_color_overlap = params['fgmodatt_color_overlap']
        fgmodatt_stimset = FGModAttendExperiment(
            mnist,
            imshape=imshape,
            bg_pix_mul=bg_pix_mul,
            texturegen=texturegen,
            texture_color_overlap=texture_color_overlap,
        )
        batch_loader = StimSetDispatcher(
            [fgmodatt_stimset, general_stimset],
            [params['p_fgmodatt_set'], 1-params['p_fgmodatt_set']],
            imshape=imshape,
        )
    else:
        batch_loader = general_stimset

    # if no texture generated added via parameters, add white noise
    if len(texturegen.generators()) == 0:
        texturegen.add_texturegen(1, defaultTexture)

    if 'max_digits' in params.keys():
        batch_loader.set_max_digits(params['max_digits'])

    if 'min_digits' in params.keys():
        batch_loader.set_min_digits(params['min_digits'])

    # Random scaling that is applied
    if 'scale_range' in params.keys():
        batch_loader.set_scale_range(params['scale_range'])
    else:
        batch_loader.set_scale_range((0.5, 1.5))

    # The probability for each digit and shape class. 
    if 'classfreq' in params.keys():
        batch_loader.set_class_freq(params['classfreq'])

    return batch_loader


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'mnist_dataset' in params.keys(
    ), 'Params must include mnist_dataset (mnist-training, mnist-validation).'

    required = ['batch_size', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print("{} initialized for dataset: {}, "
          "with bs: {}, im_shape: {}.".format(
               name,
               params['mnist_dataset'],
               params['batch_size'],
               params['im_shape']))
