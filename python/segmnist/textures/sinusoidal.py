import numpy as np
import math

from segmnist.textures.generator import TextureGenerator

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
        self._wavelength = None

    def new_example(self, ntextures):
        """ Called once for every example """
        self._ntextures = ntextures
        self._curr_texture_num = 0
        # pick a random orientation first then add random offset for each
        # foreground between 1/4 pi and 3/4 pi 
        self._orientations = [
            # background
            np.random.uniform(-math.pi/4, math.pi/4),
        ]
        # self._orientations = (
        #         np.random.choice([math.pi/4, 3*math.pi/4], size=2,
        #                          replace=False))
        # self._wavelength = np.mean(np.random.uniform(2, 6, 2))  # mean = 4
        # self._wavelength = 4
        # self._wavelength = 4 * math.sqrt(2)

    def generate(self, _=None):
        """ Called for every texture.
        The first call after new_example is assumed to the be background and
        every other call a foreground. The background is set to be a different
        orientation than the foreground.
        """
        assert self._orientations is not None, (
                'generate() must only be called after new_example()!')
        is_foreground = self._curr_texture_num > 0
        self._wavelength = np.random.uniform(2, 6) * math.sqrt(2)
        if self._curr_texture_num >= len(self._orientations):
            self._orientations.append(
                self._orientations[0] + np.random.uniform(
                    math.pi/2 - math.pi/4,
                    math.pi/2 + math.pi/4,
                ))
        orientation = self._orientations[self._curr_texture_num]
        phase = np.random.uniform(0, math.pi)
        self._curr_texture_num += 1

        # "Bounding box"
        xmax = self._shape[2]
        ymax = self._shape[1]
        xmin = 0
        ymin = 0

        (x, y) = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

        # Gratings start horizontal, then rotates clockwise
        # The value along the x of the grating is constant
        # x_rot = x * np.cos(orientation) + y * np.sin(orientation)
        y_rot = -x * np.sin(orientation) + y * np.cos(orientation)

        grat = np.cos(2 * np.pi / self._wavelength * y_rot + phase)
        clip_ratio = 1
        grat = grat / clip_ratio
        grat = np.maximum(-1.0, np.minimum(1.0, grat))
        grat = (grat + 1) / 2  # convert range from (-1, 1) to (0, 1)
        grat = grat * (self._valid_range[1] - self._valid_range[0]) - self._valid_range[0]

        # make gratings between min_brightness and max_brightness (valid_range[0] and [1])

        return(np.broadcast_to(grat, self._shape))


