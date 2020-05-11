import math

def model_factory(row):
    return ProstatexModel(row)


def image_factory(row):
    if row['Name'] == 'ktrans':
        return KtransImage(row)
    return Image(row)


def clinical_features_factory(row):
    if 'ggg' in row:
        return ClinicalFeatures2(row)
    if 'ClinSig' in row:
        return ClinicalFeatures1(row)
    return ClinicalFeatures(row)


class ClinicalFeatures:
    def __init__(self, row):
        self._fid = int(row['fid'])
        self._weight = row['PatientsWeight']
        self._size = row['PatientsSize']
        self._zone = row['zone']

    def fid(self):
        return self._fid

    def weight(self):
        return self._weight

    def size(self):
        return self._size

    def zone(self):
        return self._zone

    def as_dict(self) -> dict:
        return {
            'age': self.age(),
            'weight': self.weight(),
            'size': self.size(),
            'zone': self.zone()
        }


class ClinicalFeatures1(ClinicalFeatures):
    def __init__(self, row):
        ClinicalFeatures.__init__(self, row)
        self._significance = row['ClinSig']

    def significance(self):
        return bool(self._significance)

class ClinicalFeatures2(ClinicalFeatures):
    def __init__(self, row):
        ClinicalFeatures.__init__(self, row)
        self._significance = row['ggg']

    def significance(self):
        return int(self._significance)

class BaseImage:
    def __init__(self, row):
        self._name = row['Name']
        self._dim = row['Dim']
        self._ijk = row['ijk']
        self._id = row['ProxID']

    def id(self):
        return self._id

    def dim(self) -> list:
        return [int(v) for v in self._dim.split('x')]

    def len(self) -> int:
        return self.dim()[2]

    def spacing(self):
        return [1.5, 1.5, 3]

    # Modality name
    def name(self) -> str:
        return self._name

    def data(self):
        raise NotImplementedError  # dynamically added

    def imgdata(self):
        return self.data()[0]

    def ijk(self) -> list:
        return [int(v) for v in self._ijk.split(' ')]

    def __str__(self):
        return vars(self).__str__()


class Image(BaseImage):
    def __init__(self, row):
        BaseImage.__init__(self, row)
        self._dcm_ser_num = row['DCMSerNum']
        if math.isnan(row['DCMSerOffset']):
            self._dcm_ser_offset = 0
        else :
            self._dcm_ser_offset = int(row['DCMSerOffset'])
        self._dcm_ser_dir = str(row['DCMSerDir'])
        self._voxel_spacing = row['VoxelSpacing']

    def spacing(self):
        try:
            return [float(v) for v in self._voxel_spacing.split(',')]
        except AttributeError:
            return BaseImage.spacing(self)

    def offset(self) -> int:
        return self._dcm_ser_offset

    def dir(self) -> str:
        return self._dcm_ser_dir


class KtransImage(BaseImage):
    def __init__(self, row):
        BaseImage.__init__(self, row)

    def ijk(self) -> list:
        i, j, k = BaseImage.ijk(self)
        return [j, i, k]  # Temp fix for ROI


class ProstatexModel:
    def __init__(self, row):
        self._id = row['ProxID']
        self._clinical_features = clinical_features_factory(row)

    def id(self):
        return self._id

    def full_id(self):
        return self.id(), self.clinical_features().fid()

    def clinical_features(self) -> ClinicalFeatures:
        return self._clinical_features
