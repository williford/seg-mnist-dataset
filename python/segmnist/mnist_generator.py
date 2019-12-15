import random
import numpy as np
import scipy
import positioning as pos
from PIL import Image


def generate_digit(new_data,
                   new_segm,
                   mnist_iter,
                   texturegen,
                   mnist_shape=(28, 28),
                   scale_range=(1, 1),
                   positioning=pos.RandomPositioning):

    (nchannels, H, W) = new_data.shape
    scale = random.uniform(scale_range[0], scale_range[1])
    h = int(round(scale * mnist_shape[0]))
    w = int(round(scale * mnist_shape[1]))

    (offset_i, offset_j) = positioning.generate(
        new_data.shape[1:], (h, w))

    slice_chan = slice(nchannels)
    slice_dest_i = slice(max(0, offset_i), min(H, offset_i + h))
    slice_dest_j = slice(max(0, offset_j), min(W, offset_j + w))

# <<<<<<< HEAD
    label1, data0 = mnist_iter.__next__()
# =======
#     label1, data0 = next(mnist_iter)
# >>>>>>> Make change for python3.

    data1 = np.array(Image.fromarray(data0).resize((h, w))) # scipy.misc.imresize(data0, (h, w), 'bicubic')

    # Calculate indices within digit
    digit_offset_i = abs(min(0, offset_i))
    digit_offset_j = abs(min(0, offset_j))
    slice_src_i = slice(digit_offset_i, min(H, offset_i + h) - offset_i)
    slice_src_j = slice(digit_offset_j, min(W, offset_j + w) - offset_j)
    digit = data1[slice_src_i, slice_src_j].astype(dtype=np.float)

    digit_texture = texturegen.generate(digit / 255.0)
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
