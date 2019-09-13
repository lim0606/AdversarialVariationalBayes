"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import numpy as np
import scipy as sp
import scipy.misc
from time import gmtime, strftime
import os
from collections import defaultdict
from PIL import Image
from numpy import (amin, amax, ravel, asarray, cast, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, output_dir, image_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return imsave(inverse_transform(images), size, os.path.join(output_dir, image_name))


def save_kde_plot(z, output_dir, image_name, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = z[:, 0]
    y = z[:, 1]
    values = np.vstack([x, y])
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis(bbox)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    cset = ax.contour(xx, yy, f, colors='k')
    # ax.plot(z[:, 0], z[:, 1], 'x')

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def save_heat_map(f, output_dir, image_name, samples=None, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N, M = f.shape
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis(bbox)

    # ax.imshow(f, cmap='hot',
    #     origin='lower',
    #     extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    xx, yy = np.mgrid[bbox[0]:bbox[1]:(N*1j), bbox[2]:bbox[3]:(M*1j)]
    cfset = ax.contourf(xx, yy, f, cmap='Reds')
    cset = ax.contour(xx, yy, f, colors='k')

    if samples is not None:
        ax.plot(samples[:, 0], samples[:, 1], 'x')

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def save_z_plot(z, zlabels, output_dir, image_name, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots()

    ax.set_autoscale_on(False)
    for label in set(zlabels):
        z_i = z[zlabels==label]
        ax.plot(z_i[:, 0], z_i[:, 1], 'x')

    ax.set_aspect("equal")
    ax.axis(bbox)

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def get_bbox(samples):
    min_x, min_y = np.min(samples, axis=0)
    max_x, max_y = np.max(samples, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    center_x = (max_x + min_x)/2
    center_y = (max_y + min_y)/2
    half_size = 0.55 * max(width, height)

    bbox = [
        center_x - half_size, center_x + half_size,
        center_y - half_size, center_y + half_size
    ]

    return bbox

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    #return scipy.misc.imsave(path, merge(images, size))
    #return scipy.misc.toimage(merge(images, size), cmin=0., cmax=1.).save(path)
    return toimage(merge(images, size), cmin=0., cmax=1.).save(path)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    # return np.array(cropped_image)/127.5 - 1.
    return  np.array(cropped_image)/255.

def inverse_transform(images):
    return images
    # return (images + 1.)/2

def to_nested_dict(d):
    nested_d = defaultdict(dict)
    for (k1, k2), v in d.items():
        nested_d[k1][k2] = v
    return nested_d

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = asarray(arr)
    if iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(asarray(pal, dtype=uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (arange(0, 256, 1, dtype=uint8)[:, newaxis] *
                       ones((3,), dtype=uint8)[newaxis, :])
                image.putpalette(asarray(pal, dtype=uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = amin(ravel(data))
        if cmax is None:
            cmax = amax(ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


# Returns a byte-scaled image
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(uint8)
