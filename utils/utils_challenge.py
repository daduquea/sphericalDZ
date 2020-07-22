from __future__ import print_function

import os
from keras.utils import to_categorical
from tensorflow.python.client import device_lib

from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
#from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise,LeakyReLU,Conv2DTranspose
#from keras.layers import  Dropout,BatchNormalization,Average
from keras.layers import BatchNormalization,SpatialDropout2D
from keras.layers import Softmax, Lambda
from keras.activations import relu
from keras.optimizers import Adadelta, SGD, RMSprop
from keras.losses import binary_crossentropy
from keras.layers import GaussianNoise
import datetime
from scipy import stats
import matplotlib.pyplot as plt

        
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)                
    
    
def save_h5_file(numpy_array, file_name):
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('array',  data=numpy_array)   
        
def load_h5_file(file_name, name_dataset = 'array'):
    with h5py.File(file_name, 'r') as hf:
        data = hf[name_dataset][:]         
    return data

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']        

def get_time():
        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def load_features(base_path, file_name, channels = [0,1,2,3,4,5,6,7,8,9]):
    full_name = os.path.join(base_path, file_name)
    X_raw = np.load(full_name)
    desired_shape = compute_desired_shape(X_raw.shape)
    X_unnorm, _ = pad_array_with_zeros(X_raw, desired_shape)
    #print('xxx', X_raw.shape, X_unnorm.shape)
    # discard pixels
    # TO-DO
    #X = X_unnorm[:80,:80,:]
    X = X_unnorm
    X_mask = np.ones_like(X)
    # END discard pixels
    X_expanded = np.expand_dims(X*X_mask,axis=1)
    # move axis
    X_out = np.moveaxis(X_expanded,0,1)
    X_out = np.moveaxis(X_out,3,1)    
    return [X_out, desired_shape]


def load_gt(base_path, file_name, desired_shape, n_classes = 6, by_slices = True, class_of_interest = None):
    full_name = os.path.join(base_path, file_name)
    Y_raw = np.load(full_name)
    
    simp = np.where(Y_raw == -1)
    Y_raw[simp] = 0    
    if class_of_interest is not None:
        pixels_with_points = np.where(Y_raw > 0)
        pixels_with_class_of_interest = np.where(Y_raw == class_of_interest)
        Y_raw[pixels_with_points] = 1
        Y_raw[pixels_with_class_of_interest] = 2
        n_classes = 2 + 1 # 1 vs other classes + empty
        
    Y, _ = pad_array_with_zeros(Y_raw, desired_shape)
    
    y_low = Y[:,:,0]
    y_expanded_low = np.expand_dims(y_low, axis = 0)
    y_categ_low = to_categorical(y_expanded_low, num_classes=n_classes)
    y_categ_low = y_categ_low[:,:,:,1:]
    y_categ_low = np.moveaxis(y_categ_low,3,1)
    
    y_categ_high = None
    if by_slices:
        y_high = Y[:,:,1]
        y_expanded_high = np.expand_dims(y_high, axis = 0)
        y_categ_high = to_categorical(y_expanded_high, num_classes=n_classes)    
        y_categ_high = y_categ_high[:,:,:,1:]
        y_categ_high = np.moveaxis(y_categ_high, 3,1)
    #y_categ_concat = np.concatenate([y_categ_low, y_categ_high], axis = 0 )

    return Y, [y_categ_low, y_categ_high]

# Spherical
def load_features_spherical(base_path, file_name, channels = [0,1,2,3,4,5,6,7]):
    #feature_names = ['x', 'y', 'z', 'depth', 'abs(Nx)', 'abs(Ny)', 'abs(Nz)', 'z_angle']
    full_name = os.path.join(base_path, file_name)
    
    with h5py.File(full_name, 'r') as hf:
        X_raw = hf['array'][:]    
    X = X_raw[:,:,channels]

    # END discard pixels
    X_expanded = np.expand_dims(X,axis=1)
    # move axis
    X_out = np.moveaxis(X_expanded,0,1)
    X_out = np.moveaxis(X_out,3,1)    
    return X_out

def load_gt_spherical(base_path, file_name, n_classes = 6):
    full_name = os.path.join(base_path, file_name)
    with h5py.File(full_name, 'r') as hf:
        Y_raw = hf['array'][:]   
    #print(np.unique(Y_raw)) 
    y_expanded = np.expand_dims(Y_raw, axis = 0)
    y_categ = to_categorical(y_expanded, num_classes=n_classes)
    y_categ = y_categ[:,:,:,1:]

    return np.moveaxis(y_categ,3,1)

def generator_spherical_no_batch(path_to_X, X_list, path_to_y, y_list, variables = [0,1,2,3,4,5,6,7], n_classes = 5, is_validation = False, random_augmentation = True):
    num_samples = len(X_list)
    print('validation:', is_validation)
    print('num_samples', num_samples)
    n = 0
    while True:
        random_index = np.random.choice(num_samples)
        
        if is_validation: 
            random_index = n
        
        #print(X_list[random_index])       
        X_features = load_features_spherical(path_to_X, X_list[random_index], channels=variables)
        Y = load_gt_spherical(path_to_y, y_list[random_index])
        
        # data_augmentation    
        X_features, [Y, _] = perform_data_augmentation(is_validation, random_augmentation, X_features, Y)            
        
        n += 1
        if is_validation and n == num_samples:
            n = 0
        
        
        #for i in range(3):
        #    curr_chan = X_features[0,i,:,:]
        #    print(i, np.min(curr_chan), np.max(curr_chan))
        
        yield X_features, Y
        
def generator_spherical(path_to_X, X_list, path_to_y, y_list, batch_size = 1, variables = [0,1,2,3,4,5,6,7], n_classes = 5, is_validation = False, random_augmentation = True):
    num_samples = len(X_list)
    print('validation:', is_validation)
    print('num_samples', num_samples)
    n = 0
    all_X_features = []
    all_Y = []
    while True:
        
        random_index = np.random.choice(num_samples)
        
        if is_validation: 
            random_index = n
        
        #print(X_list[random_index])       
        X_features = load_features_spherical(path_to_X, X_list[random_index], channels=variables)
        Y = load_gt_spherical(path_to_y, y_list[random_index], n_classes = n_classes)
        
        # data_augmentation    
        X_features, [Y, _] = perform_data_augmentation(is_validation, random_augmentation, X_features, Y)            
        
        n += 1
        if is_validation and n == num_samples:
            n = 0
        
        all_X_features.append(X_features[0,:,:,:])
        all_Y.append(Y[0,:,:,:])
        #print("entra", len(all_X_features) )
        if len(all_X_features) == batch_size:
            print("returning...", np.array(all_X_features).shape, np.array(all_Y).shape)
            #yield np.array(all_X_features), np.array(all_Y)  
            all_X_features = []
            all_Y = []



## END SPHERICAL

# normals for bev
def compute_normals(height_map):
    zy, zx = np.gradient(height_map)
    normal = np.dstack((-zx, -zy, np.ones_like(height_map)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values
    normal += 1
    normal /= 2
    return normal

def compute_h_cut(data, variation = 3.):
    simp = np.ceil(data)
    z_mode = stats.mode(simp)
    cut_val = z_mode.mode[0] + variation
    return cut_val

def get_closest_multiple_of(value, multiple = 32.0):
    return int(np.ceil(value/multiple)*multiple)

def compute_desired_shape(initial_shape, multiple = 32.0):
    h_desired = get_closest_multiple_of(initial_shape[0])
    w_desired = get_closest_multiple_of(initial_shape[1])
    return (h_desired, w_desired)

def pad_array_with_zeros(array_to_pad, desired_shape):
    height_desired, width_desired = desired_shape
    if array_to_pad.ndim == 2:
        depth = 1
    else:
        depth = array_to_pad.shape[2]
    padded_array = np.zeros([height_desired, width_desired, depth])
    h_offset = (padded_array.shape[0] - array_to_pad.shape[0]) // 2
    w_offset = (padded_array.shape[1] - array_to_pad.shape[1]) // 2
    
    if array_to_pad.ndim == 2:
        padded_array[h_offset:array_to_pad.shape[0]+h_offset,w_offset:array_to_pad.shape[1]+w_offset, 0] = array_to_pad
    else:    
        padded_array[h_offset:array_to_pad.shape[0]+h_offset,w_offset:array_to_pad.shape[1]+w_offset, :] = array_to_pad
    return padded_array, [h_offset, w_offset]

def compute_global_parameters_of_proj(proj, full_xyz_points):
    ## get parameters of full image
    height_glob, width_glob = proj.projector.get_image_size(points=full_xyz_points)
    min_val_glob = full_xyz_points.min(0)
    verbose = False
    if verbose:
        print(height_glob, width_glob)
        print(min_val_glob)    
        bev_hmax_glob = proj.project_points_values(full_xyz_points, full_xyz_points[:,2], aggregate_func = 'max')
        plt.title('global')
        plt.imshow(bev_hmax_glob)
        plt.show()
    return height_glob, width_glob, min_val_glob

def perform_data_augmentation(is_validation, random_augmentation, X, Y_low, Y_high = None):
    if not is_validation and random_augmentation:
        random_aug = np.random.rand(1)
        if random_aug > 0.5:
            ax = 2 # horizontal_flip
            #ax = 3 # vertical_flip
            X = np.flip(X, axis = ax)
            Y_low = np.flip(Y_low, axis = ax)
            if Y_high is not None:
                Y_high = np.flip(Y_high, axis = ax)
        random_aug = np.random.rand(1)
        if random_aug > 0.5:
            #ax = 2 # horizontal_flip
            ax = 3 # vertical_flip
            X = np.flip(X, axis = ax)
            Y_low = np.flip(Y_low, axis = ax)
            if Y_high is not None:
                Y_high = np.flip(Y_high, axis = ax)            
            
    return X, [Y_low, Y_high]
    
# 2D
def generator(path_to_X, X_list, path_to_y, y_list, desired_shape, variables = [0,1,2,3,4,5,6,7,8,9], n_classes = 5, is_validation = False, random_augmentation = True):
    num_samples = len(X_list)
    print('validation:', is_validation)
    print('num_samples', num_samples)
    n = 0
    while True:
        random_index = np.random.choice(num_samples)
        
        if is_validation: 
            random_index = n
        
        
        [X_features, desired_shape] = load_features(path_to_X, X_list[random_index], channels=variables)
        _, [y_gt_low, y_gt_high] = load_gt(path_to_y, y_list[random_index], desired_shape)
        
        # data_augmentation
        #X_features, [y_gt_low, y_gt_high] = perform_data_augmentation(is_validation, random_augmentation, X_features, y_gt_low, y_gt_high)
        
        
        n += 1
        if is_validation and n == num_samples:
            n = 0
        yield X_features, [y_gt_low, y_gt_high]
        

def generator_single_bev(path_to_X, X_list, path_to_y, y_list, variables = [0,1,2,3,4,5,6,7,8,9], n_classes = 5, is_validation = False, random_augmentation = True, by_slices = False, class_of_interest = None, zero_pixels = False):
    
    num_samples = len(X_list)
    print('validation:', is_validation)
    print('num_samples', num_samples)
    n = 0
    while True:
        random_index = np.random.choice(num_samples)
        
        if is_validation: 
            random_index = n

        [X_features, desired_shape] = load_features(path_to_X, X_list[random_index], channels=variables)
        Y, [y_gt_low, _] = load_gt(path_to_y, y_list[random_index], desired_shape, by_slices = by_slices, class_of_interest = class_of_interest)
        
        if zero_pixels:
            Y = Y[:,:,0]
            mask = np.ones_like(Y)
            mask[Y == 0] = 0
            for i in range(X_features.shape[1]):
                X_features[0,i,:,:] = X_features[0,i,:,:] * mask
        
        # data_augmentation
        X_features, [y_gt_low, _] = perform_data_augmentation(is_validation, random_augmentation, X_features, y_gt_low, Y_high = None)
        
        
        n += 1
        if is_validation and n == num_samples:
            n = 0
        #print(X_features.shape, y_gt_low.shape)
        return X_features, y_gt_low  
        
def generator_3D(path_to_X, X_list, path_to_y, y_list, variables = [0,1,2,3,4,5,6,7,8,9], is_validation = False, random_augmentation = True, by_slices = False, class_of_interest = None, zero_pixels = False, batch_size = 1):
    
    n_features_by_slice = len(variables) // 2 # num of slices: 2 [low, high]
    num_samples = len(X_list)
    print('validation:', is_validation)
    print('num_samples', num_samples)
    n = 0
    all_X_features = []
    all_Y = []
    
    while True:
        random_index = np.random.choice(num_samples)
        
        if is_validation: 
            random_index = n

        [X_features, desired_shape] = load_features(path_to_X, X_list[random_index], channels=variables)
        Y, [y_gt_low, y_gt_high] = load_gt(path_to_y, y_list[random_index], desired_shape, by_slices = by_slices, class_of_interest = class_of_interest)
        
        if zero_pixels:
            zeroed_low_features = set_empty_pixels_to_zero(X_features[:,:len(variables)//2,:,:], Y[:,:,0])
            zeroed_high_features = set_empty_pixels_to_zero(X_features[:,len(variables)//2:,:,:], Y[:,:,1])
            X_features = np.concatenate([zeroed_low_features, zeroed_high_features], axis = 1)     
        
        # data_augmentation
        X_features, [y_gt_low, y_gt_high] = perform_data_augmentation(is_validation, random_augmentation, X_features, y_gt_low, y_gt_high)
        
        
        n += 1
        if is_validation and n == num_samples:
            n = 0
            
        #FROM 2D to 3D
        Xtemp=[]
        for i in range(n_features_by_slice):
            Xtemp.append(X_features[:,(i,i+n_features_by_slice),:,:])
        X_features=np.expand_dims(np.concatenate(Xtemp,axis=0),axis=0)
        Y_conc = np.concatenate([np.expand_dims(y_gt_low,axis=2), np.expand_dims(y_gt_high,axis=2)], axis=2)
        
        all_X_features.append(X_features[0,:,:,:,:])
        all_Y.append(Y_conc[0,:,:,:,:])
             
        if len(all_X_features) == batch_size:
            #yield np.array(all_X_features), np.array(all_Y)  
            return np.array(all_X_features), np.array(all_Y)  
            all_X_features = []
            all_Y = []

# assigning to zero empty pixels after loading files        
def set_empty_pixels_to_zero(features_array, gt_array):
    non_valid_idx = np.where(gt_array == 0)
    for i in range(features_array.shape[1]):
        features_array[0,i,non_valid_idx[0], non_valid_idx[1]] = 0    
    return features_array        
        
# revisar antes de  volver a usar    
def generator3D(path_to_X, X_list, path_to_y, y_list, desired_shape, variables = [0,1,2,3,4,5,6,7,8,9], n_classes = 5, is_validation = False, random_augmentation = True, zero_empty = False, by_slices = False, class_of_interest = None):
    num_samples = len(X_list)
    print('validation:', is_validation)
    print('num_samples', num_samples)
    n = 0
    while True:
        random_index = np.random.choice(num_samples)
        if is_validation: 
            random_index = n


        [X_features, desired_shape] = load_features(path_to_X, X_list[random_index], channels=variables)
        Y, [y_gt_low, y_gt_high] = load_gt(path_to_y, y_list[random_index], desired_shape)
        
        if zero_empty:
            zeroed_low_features = set_empty_pixels_to_zero(X_features[:,:5,:,:], Y[:,:,0])
            zeroed_high_features = set_empty_pixels_to_zero(X_features[:,5:,:,:], Y[:,:,1])
            X_features = np.concatenate([zeroed_low_features, zeroed_high_features], axis = 1)        
        
        n += 1
        if is_validation and n == num_samples: 
            n = 0
            
        # data_augmentation    
        X_features, [y_gt_low, y_gt_high] = perform_data_augmentation(is_validation, random_augmentation, X_features, y_gt_low, y_gt_high)            

        #FROM 2D to 3D
        Xtemp=[]
        for i in range(n_classes):
            Xtemp.append(X_features[:,(i,i+n_classes),:,:])
        X_features=np.expand_dims(np.concatenate(Xtemp,axis=0),axis=0)

        yield X_features, np.concatenate([np.expand_dims(y_gt_low,axis=2),np.expand_dims( y_gt_high,axis=2)], axis=2)
        
        
        
def get_output_of_layer(model, x, idx_output_layer, gt_colored_sph, pred_colored_sph, show_images = True):
    get_nth_layer_output = K.function([model.layers[0].input],
                                      [model.layers[idx_output_layer].output])
    print("layer id:", idx_output_layer)
    print("output of: " + model.layers[idx_output_layer].name)
    layer_output = get_nth_layer_output([x])[0]
    layer_output = layer_output[0,:,:,:]
    print("shape", layer_output.shape)
    if show_images:
        plt.figure(figsize=(20,20))
        plt.title("ground truth")
        plt.imshow(gt_colored_sph)
        plt.show()        
        for i in range(layer_output.shape[0]):
            plt.figure(figsize=(20,20))
            plt.title("filter: " + str(i))
            plt.imshow(layer_output[i,:,:])
            plt.show()    
        plt.figure(figsize=(20,20))
        plt.title("prediction")
        plt.imshow(pred_colored_sph)
        plt.show()
    return layer_output