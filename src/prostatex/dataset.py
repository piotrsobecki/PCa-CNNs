import pydicom
import numpy as np
import os
import pandas as pd
import re
from functools import lru_cache
from functools import partial
import logging
import prostatex.model as prostatexmodel
from os.path import exists
import prostatex.utils.mhd_utils as mhd_utils

# Utils
def ds_get(ds, key, default):
    if key in ds:
        return ds.data_element(key).value
    else:
        return default


class BaseDataSet:
    name_map = {
        't2_tse_tra_320_p20': 't2-tra',
        't2_tse_tra0': 't2-tra',
        't2_tse_tra_Grappa30':'t2-tra',
        't2_tse_cor0': 't2-cor',
        't2_tse_sag0': 't2-sag',
        't2_tse_cor_320_p20':'t2-cor',
        't2_tse_sag_320_p20': 't2-sag',
        'ep2d_diff_tra_DYNDIST_ADC0': 'dwi-adc',
        'ep2d_diff_tra_DYNDIST_MIX_ADC0': 'dwi-adc',
        'diffusie_3Scan_4bval_fs_ADC0': 'dwi-adc',
        'ep2d_diff_tra2x2_Noise0_FS_DYNDIST_ADC0': 'dwi-adc',
        'ep2d_diff_tra2x2_Noise0_NoFS_DYNDIST_ADC0': 'dwi-adc',
        'ep2d_advdiff_3Scan_4bval_spair_511b_ADC0': 'dwi-adc',
        'ep2d_advdiff_MDDW_12dir_spair_511b_ADC0': 'dwi-adc',
        'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC0': 'dwi-adc',
        'diff_tra_b_50_500_800_WIP511b_alle_spoelen_ADC0': 'dwi-adc',
        'ktrans': 'ktrans',
        'ep2d_diff_tra_DYNDISTCALC_BVAL0': 'bval',
        'ep2d_diff_tra_DYNDIST_MIXCALC_BVAL0':'bval',
        'diffusie_3Scan_4bval_fsCALC_BVAL0':'bval',
        'diff_tra_b_50_500_800_WIP511b_alle_spoelenCALC_BVAL0':'bval',
        'ep2d_advdiff_3Scan_4bval_spair_511bCALC_BVAL0':'bval',
        'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen0':'bval',
        'ep2d_diff_tra2x2_Noise0_FS_DYNDISTCALC_BVAL0':'bval'
    }

    forward_desc_name = [
        't2_tse_cor:*tse2d1_25',
        'diffusie-3Scan-4bval_fs:ep_b50t',
        'diffusie-3Scan-4bval_fs:ep_b500t',
        'diffusie-3Scan-4bval_fs:ep_b800t',
        'diffusie-3Scan-4bval_fs_ADC:ep_b50_800',
        'diffusie-3Scan-4bval_fsCALC_BVAL:ep_b50_800',
        'tfl_3d dynamisch fast:*tfl3d1',
        't2_tse_tra:*tse2d1_15'
    ]

    rel = '.'
    sep = ','
    dicom_dir = "DOI"
    ktrans_dir = "ktrans"
    ktrans_name = "ktrans"
    ktrans_image_name_format = "{0}-Ktrans.mhd"
    screenshots_dir = "screenshots"
    findings_file = "lesion-information/ProstateX-Findings.csv"
    images_file = "lesion-information/ProstateX-Images.csv"
    images_ktrans_file = "lesion-information/ProstateX-Images-KTrans.csv"
    data_file = "lesion-information/ProstateX.csv"

    current_data = None
    current_images = None
    current_findings = None

    def __init__(self, base_dir):
        self.rel = base_dir
        self.logger = logging.getLogger('prostatex')

    def get_name_mapping(self, modalities):
        return dict((k, v) for k, v in self.name_map.items() if v in modalities)

    def get_dicom_dir(self):
        return os.path.join(self.rel, self.dicom_dir)

    def get_ktrans_dir(self):
        return os.path.join(self.rel, self.ktrans_dir)

    def get_screenshot(self, fname):
        return os.path.join(self.rel, self.screenshots_dir, fname)

    def images(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.rel, self.images_file), sep=",")

    def images_ktrans(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.rel, self.images_ktrans_file), sep=",")
        df['Name'] = self.ktrans_name
        return df

    def findings(self) -> pd.DataFrame:
        findings = pd.read_csv(os.path.join(self.rel, self.findings_file), sep=",")
        findings = findings.drop_duplicates(subset=['ProxID', 'fid'], keep=False)
        return findings

    def data(self) -> pd.DataFrame:
        location = os.path.join(self.rel, self.data_file)
        if exists(location):
            data = pd.read_csv(location, sep=self.sep, index_col=None)
        else:
            data = self.merge_data()
            self.save_data(data)
        return data

    def data_aslist(self) -> list:
        data = self.data()
        group_keys = ['ProxID', 'fid']
        data_aslist = []
        for name, group in data.groupby(group_keys):
            model = prostatexmodel.model_factory(group.iloc[0])
            images_as_list = group.apply(prostatexmodel.image_factory, axis=1)
            for image in images_as_list:
                image.data = partial(self.image, model=image)
            model.images = {self.name_map[image.name()]: image for image in images_as_list if image.name() in self.name_map}
            data_aslist.append(model)
        return data_aslist

    def save_data(self, dataframe):
        dataframe.to_csv(os.path.join(self.rel, self.data_file), sep=self.sep, index=False)

    def merge_data(self) -> pd.DataFrame:
        self.logger.info('Start merging the data.')
        images = self.images()
        findings = self.findings()
        images_ktrans = self.images_ktrans()
        images_dcm = self.dcm_data()
        merged = pd.merge(images, findings, how='left', on=['ProxID', 'fid'], suffixes=('', '_y'))
        merged_all = pd.merge(merged, images_dcm, how='left', on=['ProxID', 'DCMSerNum', 'DCMSerDescr', 'Name'],
                              suffixes=('', '_y'))
        merged_ktrans = pd.merge(images_ktrans, findings, how='right', on=['ProxID', 'fid'], suffixes=('', '_y'))
        merged_all = merged_all.append(merged_ktrans)
        merged_all = merged_all.drop([col for col in merged_all.columns.values if col.endswith('_y')], axis=1)
        return merged_all

    def dcm_data(self) -> pd.DataFrame:
        d = {'ProxID': [],
             # 'PatientsAge': [],
             'PatientsWeight': [],
             'DCMSerDir': [],
             'DCMSerDescr': [],
             'DCMSerLen': [],
             'DCMSerNum': [],
             'DCMSeqName': [],
             'PatientsSize': [],
             'Name': [],
             'DCMSerOffset': []
             }
        pathdicom = self.get_dicom_dir()
        for patientDir in os.listdir(pathdicom):
            self.logger.info('Merging for: ' + patientDir)
            cur = os.path.join(pathdicom, patientDir)
            for dataDir in os.listdir(cur):
                cur = os.path.join(pathdicom, patientDir, dataDir)
                print(cur)
                for seriesDir in os.listdir(cur):
                    ser_dir = os.path.join(patientDir, dataDir, seriesDir)
                    dcm, dss, seqs = self.load_dcm(ser_dir)
                    nameSuffix = len(np.unique(seqs)) - 1
                    seqNames = []
                    for idx, ds in enumerate(dss):
                        seqName = ds_get(ds, "SequenceName", "")
                        if seqName not in seqNames and ds is not None:
                            seqNames.append(seqName)
                            serDesc = ds_get(ds, "SeriesDescription", None)
                            d['DCMSeqName'].append(seqName)
                            d['ProxID'].append(patientDir)
                            d['DCMSerDir'].append(ser_dir.replace("\\", "/"))
                            d['DCMSerDescr'].append(serDesc)
                            # d['PatientsAge'].append(int(re.search(r'\d+', ds_get(ds, "PatientsAge", "")).group()))
                            d['PatientsWeight'].append(ds_get(ds, "PatientsWeight", None))
                            pat_size = float(ds_get(ds, "PatientsSize", 0))
                            if pat_size < 3:
                                pat_size *= 100
                            d['PatientsSize'].append(int(pat_size))
                            d['DCMSerNum'].append(ds_get(ds, "SeriesNumber", None))
                            d['DCMSerLen'].append(len(dss))
                            descToName = serDesc.replace(" ", "_").replace(".", "_").replace("-", "_").replace("=", "_")
                            d['Name'].append(descToName + str(nameSuffix))
                            d['DCMSerOffset'].append(str(int(idx)))
                            nameSuffix -= 1
        return pd.DataFrame(data=d)

    #@lru_cache(maxsize=16)
    def load_dcm(self, dcm_dir, offset=0, length=-1):
        dcm_dir_corr = os.path.join(self.rel, self.dicom_dir, dcm_dir)
        dcms = os.listdir(dcm_dir_corr)
        dss = {}
        for idx, dcm in enumerate(dcms):
            ds = pydicom.read_file(os.path.join(dcm_dir_corr, dcm))
            instance_number = int(ds_get(ds, 'InstanceNumber', None))
            seqdescname = ds_get(ds, 'SeriesDescription', '') + ':' + ds_get(ds, 'SequenceName', '')
            # print(seqDescName)
            if seqdescname in self.forward_desc_name:
                z = instance_number - 1
            else:
                z = len(dcms) - instance_number
            dss[z] = ds

        if length == -1:
            length = len(dss) - offset

        dcm_mat = np.zeros((int(dss[0].Columns), int(dss[0].Rows), max(dss.keys()) + 1),
                           dtype=dss[0].pixel_array.dtype)

        seqNames = {}
        for idx, ds in dss.items():
            seqNames[idx] = ds_get(ds, 'SequenceName', ds_get(ds, 'SeriesDescription', ""))
            dcm_mat[:, :, idx] = np.array(ds.pixel_array).T

        out_dcm = dcm_mat[:, :, offset:offset + length]
        out_dss = []
        out_seq_names = []

        for idx in range(max(offset, 0), min(offset + length, len(dss))):
            if idx in dss:
                out_dss.append(dss[idx])
                out_seq_names.append(seqNames[idx])
            else:
                length = min(offset + length + 1, len(dss))  # In theory, will not have any effect though
        return out_dcm, out_dss, out_seq_names

    #@lru_cache(maxsize=16)
    def load_ktrans(self, patientID):
        ktrans_image = os.path.join(self.rel, self.ktrans_dir, patientID,
                                    self.ktrans_image_name_format.format(patientID))
        image, meta = mhd_utils.load_raw_data_with_mhd(ktrans_image)
        return image, meta

    def image(self, model: prostatexmodel.Image):
        if model.name() == self.ktrans_name:
            return self.load_ktrans(model.id())
        else:
            return self.load_dcm(model.dir(), model.offset(), model.len())



class DataSet(BaseDataSet):
    def __init__(self, base_dir):
        BaseDataSet.__init__(self, base_dir)

    def get_label_column(self):
        return 'clinsig'


class DataSet2(BaseDataSet):
    def __init__(self, base_dir):
        BaseDataSet.__init__(self, base_dir)

    def get_model(self, row):
        return Model2(row)

    def get_label_column(self):
        return 'ggg'
