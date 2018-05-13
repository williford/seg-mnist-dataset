import random
import numpy as np
import scipy


class GridPositioning(object):
    def __init__(self, grid_shape):
        """ Should be initialized once per example """
        self._grid_shape = grid_shape
        self.reset()

    def reset(self):
        self._remaining_pos = range(self._grid_shape[0]*self._grid_shape[1])

    def generate(self, canvas_shape, digit_shape, center=False, max_ratio_outside=0.25):
        pos_index = self._remaining_pos.pop()
        i = pos_index // self._grid_shape[1]
        j = pos_index % self._grid_shape[1]

        (H, W) = canvas_shape
        (h, w) = digit_shape

        offset_i = (H / 2 + i * H )/ self._grid_shape[0]
        offset_j = (W / 2 + j * W )/ self._grid_shape[1]

        if not center:
            offset_i -= h / 2
            offset_j -= w / 2

        return (offset_i, offset_j)


class RandomPositioning(object):
    def __init__(self, range_pix_i=None, range_pix_j=None):
        self._range_pix_i = range_pix_i
        self._range_pix_j = range_pix_j

    def generate(self, canvas_shape, digit_shape, center=False,
                 max_ratio_outside=0.25):
        """ param center - if true, return the position for the center of the
            shape, other return the position for the top-left.
            Useful if the shape might be rotated.
        """
        (H, W) = canvas_shape
        (h, w) = digit_shape

        # allow digits to be partially outside image
        min_pos_i = - h * max_ratio_outside
        min_pos_j = - w * max_ratio_outside
        max_pos_i = H - h + h * max_ratio_outside
        max_pos_j = W - w + w * max_ratio_outside
        if self._range_pix_i is not None:  # overrides previous
            min_pos_i = min(self._range_pix_i)
            max_pos_i = max(self._range_pix_i) + 1
        if self._range_pix_j is not None:
            min_pos_j = min(self._range_pix_j)
            max_pos_j = max(self._range_pix_j) + 1

        offset_i = random.randrange(int(round(min_pos_i)),
                                    int(round(max_pos_i)))
        offset_j = random.randrange(int(round(min_pos_j)),
                                    int(round(max_pos_j)))

        # import pdb
        # pdb.set_trace()
        if center:
            offset_i += h / 2
            offset_j += w / 2

        return (offset_i, offset_j)

    def reset(self):
        pass


class StaticPositioning(RandomPositioning):
    def __init__(self, pos_i, pos_j):
        super(self, StaticPositioning).__init__(
            range_pix_i=(pos_i, pos_i),
            range_pix_j=(pos_j, pos_j),
        )
