from generator import TextureGenerator
import numpy as np
from scipy import ndimage


class FGModTexture(TextureGenerator):
    """ Oriented texture similar to that used in figure-ground modulation
    literature.
    """
    def __init__(self, shape, valid_range=(0, 255)):
        self._shape = shape
        self._valid_range = np.array(valid_range)
        self._ntextures = None
        self._curr_texture_num = 0
        self._linelen = 10
        self._slopes = None
        self._vert = None
        self._numlines = 50 * (shape[0] * shape[1] / self._linelen)
        self._step = 3
        self._step = 1

    def new_example(self, ntextures):
        """ Called once for every example """
        self._ntextures = ntextures
        self._curr_texture_num = 0

        self._vert = np.random.choice([0, 1], size=2, replace=False)
        ind = np.random.choice([0, 1], size=2, replace=False)
        self._intensities = np.stack([
            self._valid_range[ind],
            self._valid_range[1-ind]])
        low = np.min(self._valid_range)
        high = np.max(self._valid_range)
        mid1 = low * .55 + high * .45
        mid2 = low * .45 + high * .55
        self._colors = np.stack([
            np.random.uniform(low, mid1, size=3),
            np.random.uniform(mid2, high, size=3),
        ])
        if (np.linalg.norm(self._colors[0] - self._colors[1]) <
                abs(self._valid_range[0] - self._valid_range[1]) * 0.05):
            import pdb
            pdb.set_trace()

    def generate(self, mask=None):
        """ Called for every texture.
        The first call after new_example is assumed to the be background and
        every other call a foreground. The background is set to be a different
        orientation than the foreground.

        Args:
            mask: mask for the texture (or None, for background). Does not
                account for occlusion.
        """
        assert self._vert is not None, (
                'generate() must only be called after new_example()!')
        if mask is not None:
            num_pixels = np.sum(
                ndimage.morphology.binary_erosion(mask >= 1))
        else:
            num_pixels = self._shape[1] * self._shape[2]

        vert = self._vert[min(self._curr_texture_num, 1)]

        if num_pixels * 32 < self._shape[1] * self._shape[2]:
            color = np.random.uniform(self._valid_range[0],
                                      self._valid_range[1], size=3)
            texture = np.ones(self._shape) * color.reshape([3, 1, 1])
            return texture

        color = np.random.uniform(self._colors[0],
                                  self._colors[1])
        texture = np.ones(self._shape) * color.reshape([3, 1, 1])

        # "Bounding box"
        xmax = self._shape[2]
        ymax = self._shape[1]
        xmin = 0
        ymin = 0

        (x, y) = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

        # centers
        if vert:
            x0 = np.random.choice(
                np.arange(xmin, xmax, self._step),
                size=self._numlines, replace=True)
            y0 = np.random.choice(
                np.arange(ymin - self._linelen, ymax, self._step),
                size=self._numlines, replace=True)
            x1 = x0
            y1 = y0 + self._linelen
        else:  # horizontal
            x0 = np.random.choice(
                np.arange(xmin - self._linelen, xmax, self._step),
                size=self._numlines, replace=True)
            y0 = np.random.choice(
                np.arange(ymin, ymax, self._step),
                size=self._numlines, replace=True)
            x1 = x0 + self._linelen
            y1 = y0

        for i in range(self._numlines):
            win = ((x >= min(x0[i], x1[i])) &
                   (x <= max(x0[i], x1[i])) &
                   (y >= min(y0[i], y1[i])) &
                   (y <= max(y0[i], y1[i])))

            color = np.random.uniform(self._colors[0],
                                      self._colors[1]).reshape([3, 1])
            texture[:, win] = color

        self._curr_texture_num += 1
        return(texture)
