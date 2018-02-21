from generator import TextureGenerator
import numpy as np
import math
import ipdb as pdb

class SinusoidalGratings(TextureGenerator):
    """ Texture with sinusoidal gratings.
    """
    def __init__(self, shape, freq_dist=None, valid_range=(0, 255)):
        self._freq_dist = freq_dist
        self._shape = shape
        self._valid_range = valid_range
        self._ntextures = None
        self._curr_texture_num = 0
        self._orientations = None
        self._max_or = 2  # maximum number of orientations

    def new_example(self, ntextures):
        """ Called once for every example """
        self._ntextures = ntextures
        self._curr_texture_num = 0
        # pick a random orientation first then add offset
        # orientations between 0 and pi
        self._orientations = (
                np.random.choice([math.pi/4, 3*math.pi/4], size=2, replace=False) +
                np.random.uniform(-math.pi/4,math.pi/4))

    def generate(self):
        """ Called for every texture.
        The first call after new_example is assumed to the be background and
        every other call a foreground. The background is set to be a different
        orientation than the foreground.
        """
        assert self._orientations is not None, (
                'generate() must only be called after new_example()!')
        orientation = self._orientations[min(self._curr_texture_num, 1)]
        phase = np.random.uniform(0, math.pi)
        wavelength = np.random.uniform(2, 8)
        # def random_color_texture(self, shape, mean, var):
        self._curr_texture_num += 1

        # "Bounding box"
        xmax = self._shape[2]
        ymax = self._shape[1]
        xmin = 0
        ymin = 0

        (x, y) = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

        # Rotation (around center of image) - doesn't this assume center is 0??
        # The value along the x of the grating is constant
        y_theta = (x * np.sin(orientation) +
                   y * np.cos(orientation))

        # initially make gratings vary from 0 to 1
        gratings = (np.cos(2 * np.pi / wavelength * y_theta + phase) + 1) / 2
        # make gratings between min_brightness and max_brightness (valid_range[0] and [1])
        gratings = gratings * (self._valid_range[1] - self._valid_range[0]) - self._valid_range[0]
        return(np.broadcast_to(gratings, self._shape))


