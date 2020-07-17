from prostatex.dataset import DataSet
from prostatex.batch import get_xy
from constants import Constants, NetStructure, Variables
from cnn.network_input import NetworkInput
import pickle
from prostatex.data_augmentation import augment_data, crop_data


class NetworkInputProvider:
    data_dir = str
    data = None
    net_structure = NetStructure
    file_prefix = str

    def __init__(self, dataset_dir: str, file_prefix: str):
        self.data_dir = dataset_dir
        self.file_prefix = file_prefix
    
    def __get_data_from_csv(self):
        dataset = DataSet(self.data_dir)
        self.data = dataset.data_aslist()

    def file_name(self, ni_name: str):
        return Constants.net_inputs_dir + "_" + ni_name + "_" + self.file_prefix + ".bin"

    def __deserialize_data(self,  name) -> NetworkInput:
        with open(self.file_name(name), mode='rb') as binary_file:
            ni = pickle.load(binary_file)
        return ni

    def __serialize_data(self, ni: NetworkInput):
        with open(self.file_name(ni.name), mode='wb') as binary_file:
            pickle.dump(ni, binary_file, protocol=4)

    def get_network_input(self, name,  discard_corrupted=True, do_preprocessing = False, do_augment=False) -> NetworkInput:
        do_serialize = True
        if not do_preprocessing:
            do_serialize = False
            print("DESERIALIZING: %s" % name)
            ni = self.__deserialize_data(name)
            print("DESERIALIZED")
        else:
            print("PROCESSING: %s" % name)
            if self.data is None:
                self.__get_data_from_csv()
            roi_width = Variables.roi_width
            roi_depth = Variables.roi_depth
            ni = NetworkInput(name, *get_xy(self.data, roi_width, roi_depth, discard_corrupted=discard_corrupted ))
            if do_augment:
                ni = self.do_augment(name, ni, Variables.width_crop, Variables.depth_crop, do_serialize=False)
            #else:
            #    ni = crop_data(ni, Variables.width_crop, Variables.depth_crop)
            print("PROCESSED: %s" % name)

        if do_serialize:
            print("SERIALIZING: %s" % name)
            self.__serialize_data(ni)
            print("SERIALIZED")
        ni.describe()
        return ni


    def do_augment(self, name, train_input:NetworkInput, width_crop, depth_crop, do_serialize=True):
        ni = train_input.copy()
        ni.name = name
        ni = augment_data(name, ni, width_crop, depth_crop)
        print("PROCESSED: %s" % name)
        if do_serialize:
            print("SERIALIZING: %s" % name)
            self.__serialize_data(ni)
            print("SERIALIZED")
        return ni
