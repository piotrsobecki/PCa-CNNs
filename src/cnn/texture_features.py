import numpy
import mahotas
from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix
from skimage.feature import greycoprops


def stats(reg):
    return {
        "p05": numpy.percentile(reg, 5),
        "p10": numpy.percentile(reg, 10),
        "p25": numpy.percentile(reg, 25),
        "p75": numpy.percentile(reg, 75),
        "p90": numpy.percentile(reg, 90),
        "p95": numpy.percentile(reg, 95),
        "avg": numpy.average(reg),
        "std": numpy.std(reg.ravel()),
        "skw": skew(reg.ravel()),
        "krt": kurtosis(reg.ravel())
    }


def glcm(reg, **args):
    reg = reg.astype(int)
    return greycomatrix(reg, **args)


def glcm_props(reg, features, **args):
    glcm_calc = glcm(reg, **args)
    features_out = {}
    for greycoprop in features:
        features_out[greycoprop] = greycoprops(glcm_calc, greycoprop)[0, 0]
    return features_out

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def haralick(reg, reduction):
    reg = reg.astype(numpy.int32)
    mhf = mahotas.features.haralick(reg, compute_14th_feature=False)
    if reduction is not None:
        mhf = reduction(mhf, axis=0)
    return mhf
