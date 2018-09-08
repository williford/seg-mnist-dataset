import numpy as np


class StimSetDispatcher(object):
    def __init__(self,
                 stimsets,
                 stimset_probs,
                 imshape,
                 ):
        assert len(stimsets) == len(stimset_probs)
        self._stimsets = stimsets
        self._stimset_probs = stimset_probs
        self._imshape = imshape

    def set_min_digits(self, val):
        for stimset in self._stimsets:
            stimset.set_min_digits(val)

    def set_max_digits(self, val):
        for stimset in self._stimsets:
            stimset.set_max_digits(val)

    def class_names(self):
        for stimset in self._stimsets:
            return stimset.class_names()

    def set_nchannels(self, nchannels):
        for stimset in self._stimsets:
            stimset.set_nchannels(nchannels)

    def set_scale_range(self, scale_range):
        for stimset in self._stimsets:
            stimset.set_scale_range(scale_range)

    def set_bg_pix_mul(self, bg_pix_mul):
        """ Replaces seg_prob_mask_bg"""
        for stimset in self._stimsets:
            stimset.set_bg_pix_mul(bg_pix_mul)

#     """ Return single example with image, class labels, and
#         segmentation labels.
#     """
#     def create_example(self):
#         (img_data, cls_label, seg_label) = self.create_batch(1)
#         img_data = img_data.reshape(img_data.shape[1:])
#         seg_label = seg_label.reshape(seg_label.shape[2:])
#         cls_label = cls_label.reshape(cls_label.shape[:2])
#         return (img_data, cls_label, seg_label)

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
        cls_label = np.zeros((batch_size,
                              12,  # hard-coded - may need to be modifiable in the future
                              # 10 + len(self._shapeGenerators),
                              1, 1), dtype=np.uint8)

        sel = np.random.choice(
            self._stimsets, size=batch_size, replace=True, p=self._stimset_probs)

        for i in range(batch_size):
            img, cls, seg = sel[i].create_batch(1)
            img_data[i] = img[0]
            cls_label[i] = cls[0]
            seg_label[i] = seg[0]

        return (img_data, cls_label, seg_label)

    def set_class_freq(self, freq):
        pass
