import os
import numpy as np
from data.base import IODataset
from scipy.io import loadmat

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IndustRobo')

def create_industrobo_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, seq_stride=None, sample_rate=1, file_name=0, **kwargs):
    ith_dataset = kwargs["ith_round"]
    if file_name == 0:
        file_name_train = os.path.join(_DATA_DIR, "forward_identification_without_raw_data_shifted.mat")
        df_industRobo = loadmat(file_name_train)
        seq_len_train -= 1
        seq_len_val -= 1
        seq_len_test -= 1
    else:
        file_name_train = os.path.join(_DATA_DIR, "1010m_vary_sim_position_noiseTorque{}.mat".format(ith_dataset))
        df_industRobo = loadmat(file_name_train)

    print(file_name_train)

    u_train = df_industRobo["u_train"]
    u_val   = df_industRobo["u_val"]
    u_test  = df_industRobo["u_test"]
    y_train = df_industRobo["y_train"] / 180 * np.pi
    y_val   = df_industRobo["y_val"]   / 180 * np.pi
    y_test  = df_industRobo["y_test"]  / 180 * np.pi

    if sample_rate > 0.1:
        if_downsam = True
    else:
        if_downsam = False

    if if_downsam:
        u_train = np.asarray(u_train).T[::10]
        y_train = np.asarray(y_train).T[::10]
        u_val   = np.asarray(u_val).T[::10]
        y_val   = np.asarray(y_val).T[::10]
        u_test  = np.asarray(u_test).T[::10]
        y_test  = np.asarray(y_test).T[::10]
        seq_len_train = int(seq_len_train / 10)
        seq_len_val   = int(seq_len_val / 10)
        seq_len_test  = int(seq_len_test / 10)
    else:
        u_train = np.asarray(u_train).T
        y_train = np.asarray(y_train).T
        u_val   = np.asarray(u_val).T
        y_val   = np.asarray(y_val).T
        u_test  = np.asarray(u_test).T
        y_test  = np.asarray(y_test).T

    if bool(kwargs) and ("k_max_train" in kwargs):
        k_max_train = kwargs['k_max_train']
        print("k_max_train is: ", k_max_train)
    else:
        k_max_train = 1

    if kwargs["train_rounds"] > 1:
        length_pc = (1 - k_max_train) / (kwargs["train_rounds"] - 1)
    else:
        length_pc = (1 - k_max_train) / 1

    start  = int(ith_dataset * length_pc * u_train.shape[0])
    length = int(u_train.shape[0] * k_max_train)
    u_train = u_train[start:start + length]
    y_train = y_train[start:start + length]
    print("starts at and length is : ", ith_dataset * length_pc * 100, k_max_train * 100)
    print("start,start+length : ", start, start + length)

    dataset_train = IODataset(u_train, y_train, seq_len_train, seq_stride)
    dataset_val   = IODataset(u_val,   y_val,   seq_len_val,   seq_stride)
    dataset_test  = IODataset(u_test,  y_test,  seq_len_test,  None)

    print("dataset_train.u.shape, dataset_train.y.shape")
    print(dataset_train.u.shape, dataset_train.y.shape)
    print("dataset_val.u.shape, dataset_val.y.shape")
    print(dataset_val.u.shape, dataset_val.y.shape)
    print("dataset_test.u.shape, dataset_test.y.shape")
    print(dataset_test.u.shape, dataset_test.y.shape)
    return dataset_train, dataset_val, dataset_test
