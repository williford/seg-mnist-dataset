import numpy as np


class StimSetDispatcher(object):
    def __init__(self,
                 stimsets,
                 stimset_probs,
                 ):
        assert len(stimsets) == len(stimset_probs)
        self._stimsets = stimsets
        self._stimset_probs = stimset_probs

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
        sel = np.random.choice(
            self._stimsets, size=batch_size, replace=True, p=self._stimset_probs)
        import pdb
        pdb.set_trace()

        return (img_data, cls_label, seg_label)

    def set_class_freq(self, freq):
        pass
