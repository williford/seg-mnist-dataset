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
    def generate(self, canvas_shape, digit_shape, center=False, max_ratio_outside=0.25):
        (H, W) = canvas_shape
        (h, w) = digit_shape

        # allow digits to be partially outside image
        min_pos_i = - h * max_ratio_outside
        min_pos_j = - w * max_ratio_outside
        max_pos_i = H - h + h * max_ratio_outside
        max_pos_j = W - w + w * max_ratio_outside
        offset_i = random.randrange(int(round(min_pos_i)), int(round(max_pos_i)))
        offset_j = random.randrange(int(round(min_pos_j)), int(round(max_pos_j)))

        if center:
            offset_i += h / 2
            offset_j += w / 2

        return (offset_i, offset_j)

    def reset(self):
        pass

def generate_digit(new_data,
                   new_segm,
                   mnist_iter,
                   random_texture,
                   mnist_shape=(28, 28),
                   scale_range=(1, 1),
                   positioning=RandomPositioning):

    (nchannels, H, W) = new_data.shape
    scale = random.uniform(scale_range[0], scale_range[1])
    h = int(round(scale * mnist_shape[0]))
    w = int(round(scale * mnist_shape[1]))

    (offset_i, offset_j) = positioning.generate(new_data.shape[1:], (h,w))

    slice_chan = slice(nchannels)
    slice_dest_i = slice(max(0, offset_i), min(H, offset_i + h))
    slice_dest_j = slice(max(0, offset_j), min(W, offset_j + w))

    label1, data0 = next(mnist_iter)

    data1 = scipy.misc.imresize(data0, (h, w), 'bicubic')

    # Calculate indices within digit
    digit_offset_i = abs(min(0, offset_i))
    digit_offset_j = abs(min(0, offset_j))
    slice_src_i = slice(digit_offset_i, min(H, offset_i + h) - offset_i)
    slice_src_j = slice(digit_offset_j, min(W, offset_j + w) - offset_j)
    digit = data1[slice_src_i, slice_src_j].astype(dtype=np.float)

    digit_texture = random_texture(
        (nchannels, h, w),
        mean=np.random.randint(256, size=nchannels),
        var=np.random.gamma(1, 25, size=nchannels)
    )
    digit_texture = digit_texture[slice_chan, slice_src_i, slice_src_j]

    new_data[slice_chan, slice_dest_i, slice_dest_j] = (
        np.multiply(digit / 255.0, digit_texture) +
        np.multiply((255.0 - digit) / 255.0,
                    new_data[slice_chan, slice_dest_i, slice_dest_j])
    )

    # segmentation data
    new_segm[slice_dest_i, slice_dest_j][digit > 159] = label1 + 1

    # mask out intermediate values
    new_segm[slice_dest_i, slice_dest_j][
        np.logical_and(digit > 95,
                        digit <= 159)] = 255

    # new_data and new_segm are modified in-place
    return label1
