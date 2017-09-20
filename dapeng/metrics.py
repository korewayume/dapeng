# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K


def jaccard_coefficient(output, target, axis=None, smooth=1e-5):
    """Jaccard coefficient for comparing the similarity of two
    batch of data, usually be used for binary image segmentation.
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both output and target are empty, it makes sure dice is 1.
        If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> a = tf.constant([[[[1],[0]],[[1],[1]]]],dtype=tf.float64)
    >>> b = tf.constant([[[[1],[1]],[[1],[1]]]],dtype=tf.float64)
    >>> val = jaccard_coefficient(a,b)
    >>> session = tf.Session()
    >>> session.run(val)

    References
    -----------
    - `Jaccard coefficient <https://en.wikipedia.org/wiki/Jaccard_index>`_
    """
    if axis is None:
        axis = [1, 2, 3]
    inse = K.sum(output * target, axis=axis)
    l = K.sum(output * output, axis=axis)
    r = K.sum(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    return K.mean(dice)
