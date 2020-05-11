import numpy as np
from numpy import take
from functools import reduce
from prostatex.preprocessing import get_roi_in_mm,  pad_zero

'''
Return biggest shape from supplied by volume
'''
def get_biggest_shape(shapes):
    biggest_shape = None
    biggest_shape_volume = None
    for shape in shapes:
        shape_volume = int(reduce(lambda x, y: x*y,shape)) 
        if biggest_shape is None or biggest_shape_volume < shape_volume:
            biggest_shape = shape
            biggest_shape_volume = shape_volume
    return biggest_shape

'''
Get data (features/image) for model
Given that current method uses 3 modalities, output will be 4 dimentional [x,y,z,m] - where:
- x,y are widht/height corresponding to i and j 
- z is depth corresponding to k
- m is index of modality

To match the dimensions between modalities, data is zero-padded to the biggest shape
'''
def get_model_x(model, modality, roi_width_mm, roi_depth_mm, do_interpolate=False):
    #Get region of interest of three modalities zero padded to biggest shape
    model_roi = get_roi_in_mm(model, modality, roi_width_mm, roi_depth_mm, do_interpolate)
    if model_roi is not None:
        model_x = np.zeros([*model_roi.shape, 1])
        model_x[:, :, :, 0] = model_roi
        return model_x
    return None


# Return one hot representation given possible_values
def one_hot(value, possible_values):
    return np.eye(len(possible_values))[value]

# Return one hot representation given possible_values
def one_hot_str(value, possible_values):
    return np.eye(len(possible_values))[value]

'''
Get batch from the data
Given that current method uses 3 modalities

X output will be 5 dimentional [i,x,y,z,m] - where:
- x,y are width/height corresponding to i and j 
- z is the depth corresponding to k
- m is the index of modality
- i is the index of the single model features

Y output (labels) is 2 dimentional because currently network uses one-hot representation for true/false labels (to be corrected)
'''  
def get_batch(data, batch_len, batch_num, roi_width, roi_depth, discard_corrupted: bool):
    batch_num  = int(batch_num % (len(data) / batch_len))
    start = max(0,batch_len * batch_num)
    end = min(len(data), start + batch_len)
    batch = data[start:end]
    x_additional_values = np.array(['TZ', 'PZ', 'AS', 'SV', 'CG'])
    significance_values = [True, False]
    x = None
    x_location = np.zeros([len(batch), len(x_additional_values)])
    y = np.zeros([len(batch), len(significance_values)])
    corrupted_data_indices = []
    ids = np.empty(shape=(len(batch), 2),dtype=np.object)

    modalities = ['t2-tra', 't2-sag', 't2-cor', 'dwi-adc', 'bval', 'ktrans']

    modalities_data = {}
    shapes = {}
    for i in range(len(batch)):
        model = batch[i]
        print("Processing: %s-%d  (%d/%d)" % (*model.full_id(), i+1, len(batch)))
        ids[i] = model.full_id()
        model_modalities = {name: get_model_x(model, name, roi_width, roi_depth, False) for name in modalities}

        if any([val is None for name, val in model_modalities.items()]):
            corrupted_data_indices.append(i)

        if i == 0:
            shapes = {name: val.shape for name,val in model_modalities.items()}
            modalities_data = {name: np.zeros([len(batch), *shape]) for name, shape in shapes.items() }

        for name, model_val in model_modalities.items():
            if model_val is not None:
                modalities_data[name][i, :, :, :, :] =  pad_zero(model_val, shapes[name])

        x_location[i,:] = np.array(x_additional_values == model.clinical_features().zone()).astype(int)

        if hasattr(model.clinical_features(), 'significance'):
            y[i,:] = one_hot(int(model.clinical_features().significance()), significance_values)

    model = {
        'ADC': modalities_data['dwi-adc'],
        'DWI': modalities_data['bval'],
        'T2-TRA': modalities_data['t2-tra'],
        'T2-SAG': modalities_data['t2-sag'],
        'T2-COR': modalities_data['t2-cor'],
        'DCE': modalities_data['ktrans'],
        'location': x_location
    }
    return ids, model, y


'''
Return batch data and labels from data 
'''
def get_batch_xy(data_x, data_x_additional, data_y, batch_len, batch_num):
    data_len = len(data_x)
    indexes = range(data_len)
    start = max(0, batch_len * batch_num)
    end = min(data_len, start + batch_len)
    indexes = indexes[start:end]
    x = take(data_x, indexes, 0)
    x_additional = take(data_x_additional, indexes, 0)
    y = take(data_y, indexes, 0)
    return x, x_additional, y

def get_batch_idx(data_len, batch_len, batch_num):
    start = max(0, batch_len * batch_num)
    end = min(data_len, start + batch_len)
    return [*range(start, end)]
'''
Get data and labels for all entries in data
'''
def get_xy(data, roi_width ,roi_depth,  discard_corrupted: bool):
    return get_batch(data, len(data), 0, roi_width, roi_depth, discard_corrupted)
