from enum import Enum
import os

class NetStructure(Enum):
    CNN_VGG_SIMPLE = 1
    CNN_VGG_MODALITIES = 2
    CNN_VGG_PIRADS = 3

class Variables:
    roi_width = 30 # ROI width either in mm or voxels depends on InputPreprocessing type
    roi_depth = 30 # ROI depth either in mm or voxels depends on InputPreprocessing type
    width_crop = 6
    depth_crop = 1

class Constants:
    test_name = "CNN_VGG_PIRADS"
    test_data_name = "cnn-m5"
    architecture = NetStructure.CNN_VGG_PIRADS
    cv_folds = 5
    do_serialization = True
    do_train = True
    save_step = False
    min_epochs = 1000
    max_epochs = 10200
    checkpoint_epochs = [2500, 5000, 10000]#, 25000]
    data_augment_approx = 15 * 324
    batch_size = 32
    model_to_load = str
    saved_models_count = 3
    rel = str # project relative path
    device_type = "/gpu:0"  # Tensorflow gpu/cpu usage
    models_dir = "nets/"  # Trained nets directory
    net_inputs_dir = "nets/inputs/"  # Serialized nets input directory
    log_dir = "log/"  # Serialized nets input directory
    dataset_dir = str  # Training dataset directory
    dataset_test_dir = str # Test dataset directory
#1 3
    def __init__(self):
        Constants.rel = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
        Constants.dataset_dir = "D:\workspace\github\phd\data\ProstateX/train"
        Constants.dataset_test_dir = "D:\workspace\github\phd\data\ProstateX/test"
        Constants.models_dir = Constants.rel + Constants.models_dir
        Constants.net_inputs_dir = Constants.rel + Constants.net_inputs_dir
        Constants.log_dir = Constants.rel + Constants.log_dir

