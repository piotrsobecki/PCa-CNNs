import numpy


# Normalization functions
class NormalizationNo():
    def normalize(self, img, settings=None):
        if settings is None:
            settings = {}
        return img


class NormalizationMean(NormalizationNo):
    def normalize(self, img, settings=None):
        if settings is None:
            settings = {}
        if img.std() == 0:
            return img
        return (img - img.mean()) / img.std()


class NormalizationMedian(NormalizationNo):
    def normalize(self, img, settings=None):
        if settings is None:
            settings = {}
        denominator = numpy.median(img) + 2 * img.std()
        if denominator == 0.0:
            return img
        return img / denominator

class NormalizationFeatureScaling(NormalizationNo):

    def __init__(self, vmin=0, vmax=1):
        self.vmin=vmin
        self.vmax=vmax

    def normalize(self, img, settings=None):
        if settings is None:
            settings = {}
        OldValue = img
        OldMin = img.min()
        OldMax = img.max()
        NewMax = self.vmax
        NewMin = self.vmin
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        if OldRange == 0.0:
            return img
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        return NewValue
