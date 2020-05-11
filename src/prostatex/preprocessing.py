import numpy
from numpy import pad
import math

from constants import Variables
from prostatex.model import Image
from prostatex.normalization import NormalizationMean, NormalizationFeatureScaling, NormalizationMedian
from scipy.ndimage.interpolation import zoom

def roi_mm(image: Image, image_data: numpy.ndarray, ij_width: float, k_width: float, do_interpolate=False):
    i, j, k = image.ijk()
    i, j, k = i, j, k
    ivs, jvs, kvs = image.spacing()
    i_margin = int(math.ceil(ij_width / (2 * ivs))) + Variables.width_crop
    j_margin = int(math.ceil(ij_width / (2 * jvs))) + Variables.width_crop
    k_margin = int(math.floor(k_width / (2 * kvs))) + Variables.depth_crop
    image_data = pad(image_data, ((i_margin, i_margin), (j_margin, j_margin), (k_margin, k_margin)), mode='constant')
    img_shape = image_data.shape
    i_from = max(0, i)
    i_to = min(img_shape[0], i + 2 * i_margin)
    j_from = max(0, j)
    j_to = min(img_shape[1], j + 2 * j_margin)
    k_from = max(0, k)
    k_to = min(img_shape[2], k + 2 * k_margin +1)
    if k_margin == 0:
        roi_arr = image_data[i_from:i_to, j_from:j_to,  min(img_shape[2]-1,k_from)]
        return roi_arr[:,:,None] # add additional dimension to array
    return image_data[i_from:i_to, j_from:j_to, k_from:k_to]

def clean_outliers(image_img):
    perc_bounds = numpy.percentile(image_img, [1, 99])
    p_low = perc_bounds[0]
    p_high = perc_bounds[1]
    image_img[image_img <= p_low] = p_low
    image_img[image_img >= p_high] = p_high
    return image_img

def get_roi_in_mm(model,modality,ij_width_mm=30,k_width_mm=0, do_interpolate=False):
    try:
        if modality in model.images:
            image = model.images[modality]
            image_img = image.imgdata()
            image_img = clean_outliers(image_img)
            #image_img = NormalizationMean().normalize(image_img)
            image_img = NormalizationMedian().normalize(image_img)
            image_img = NormalizationFeatureScaling(vmin=0.0,vmax=1.0).normalize(image_img)
            img = roi_mm(image,image_img,ij_width_mm,k_width_mm, do_interpolate)
            if do_interpolate:
                zoom_factor = [ z/min(image.spacing()) for z in image.spacing()]
                img = zoom(img, zoom_factor)
            return img
    except Exception as e:
        print(model.id(), modality, e)
    return None

def pad_zero(matrix, target_shape):
    pads = []
    matrix_update = matrix.copy()
    if matrix is None:
        return numpy.zeros(shape=target_shape)
    for n in range(len(matrix.shape)):
        dim = matrix.shape[n]
        target_dim = target_shape[n]
        dim_pad_lo = int(numpy.floor((target_dim-dim)/2))
        dim_pad_hi = int(numpy.ceil((target_dim-dim)/2))
        if dim_pad_lo < 0 or dim_pad_hi < 0:
            from_lo = abs(dim_pad_lo)
            from_high = abs(dim_pad_hi)
            indices = range(0,dim)[from_lo:-from_high]
            matrix_update = numpy.take(matrix_update,indices,axis=n)

    #print('pad_zero',matrix.shape,target_shape)
    for n in range(len(matrix_update.shape)):
        dim = matrix_update.shape[n]
        target_dim = target_shape[n]
        dim_pad_lo = int(numpy.floor((target_dim-dim)/2))
        dim_pad_hi = int(numpy.ceil((target_dim-dim)/2))
        pads.append((dim_pad_lo, dim_pad_hi))
    return pad(matrix_update, pads, mode='constant')
