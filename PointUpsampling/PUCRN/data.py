import torch
import torch.utils.data as data
import h5py
import re
import os
from math import log
import numpy as np
import copy
import random

from utils import pc_utils
from network.operations import group_knn

class H5Dataset(data.Dataset):
    """
    load the entire hdf5 file to memory
    """

    def __init__(self, h5_path, num_shape_point, num_patch_point,
                 phase="train",
                 up_ratio=4, step_ratio=4,
                 jitter=False, jitter_max=0.03, jitter_sigma=0.01,
                 batch_size=16, drop_out=1.0, use_random_input= True):
        super(H5Dataset, self).__init__()
        np.random.seed(0)
        self.phase = phase
        self.batch_size = 1
        self.num_patch_point = num_patch_point
        self.num_shape_point = num_shape_point
        self.jitter = jitter
        self.jitter_max = jitter_max
        self.jitter_sigma = jitter_sigma
        self.step_ratio = step_ratio
        self.use_random_input = use_random_input
        self.input_array, self.label_array, self.data_radius = self.load_h5_data(
            h5_path, up_ratio, step_ratio, num_shape_point)
        self.sample_cnt = self.input_array.shape[0]

        self.curr_scales = [step_ratio **
                            r for r in range(1, int(log(up_ratio, step_ratio))+1)]

    def __len__(self):
        return self.sample_cnt // self.batch_size

    def load_h5_data(self, h5_path, up_ratio, step_ratio, num_point):
        # print("========== Loading Data ==========")
        num_4X_point = int(num_point * 4)
        num_out_point = int(num_point * up_ratio)

        skip_rate = 1

        print("loading data from: {}".format(h5_path))
        if self.use_random_input:
            print("use random input")
            with h5py.File(h5_path, 'r') as f:
                input_data = f['poisson_%d' % num_4X_point][:]
                gt = f['poisson_%d' % num_out_point][:]
        else:
            print("Do not use random input_data")
            with h5py.File(h5_path, 'r') as f:
                input_data = f['poisson_%d' % num_point][:]
                gt = f['poisson_%d' % num_out_point][:]

        # name = f['name'][:]
        assert len(input_data) == len(gt)
        print("Normalize the data")
        data_radius = np.ones(shape=(len(input_data)))
        centroid = np.mean(input_data[:, :, 0:3], axis=1, keepdims=True)
        input_data[:, :, 0:3] = input_data[:, :, 0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(input_data[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True) 
        input_data[:, :, 0:3] = input_data[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
        gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        
        input_data = input_data[::skip_rate]
        gt = gt[::skip_rate]
        data_radius = data_radius[::skip_rate]

        ## shuffle data 
        if self.phase == "train":
            idx = np.arange(input_data.shape[0])
            random.shuffle(idx)
            input_data = input_data[idx, ...]
            gt = gt[idx, ...]

        label = {}
        label["x%d" % up_ratio] = gt

        return input_data, label, data_radius


    def augment(self, input_patches, label_patches):
        """
        data augmentation
        """
        # random rotation
        input_patches, label_patches = pc_utils.rotate_point_cloud_and_gt(
            input_patches, label_patches)

        # jitter perturbation 
        if self.jitter:
            input_patches = pc_utils.jitter_perturbation_point_cloud(
                input_patches, sigma=self.jitter_sigma, clip=self.jitter_max)

        return input_patches, label_patches

    def __getitem__(self, index):

        ratio = self.curr_scales[np.random.randint(len(self.curr_scales))]
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        input_patches = self.input_array[start_idx:end_idx, ...]
        label_patches = self.label_array["x%d" % ratio][start_idx:end_idx, ...]
        data_radius = self.data_radius[start_idx:end_idx, ...]
    

        if self.use_random_input:
            new_batch_input = np.zeros((self.batch_size, self.num_patch_point, input_patches.shape[2]), dtype=np.float32)
            for i in range(self.batch_size):
                idx = pc_utils.nonuniform_sampling(input_patches.shape[1], sample_num=self.num_patch_point)
                new_batch_input[i, ...] = input_patches[i][idx]
            input_patches = new_batch_input[:, :, :3]

        # augment data
        if self.phase == "train":
            input_patches, label_patches = self.augment(input_patches, label_patches)
            data_radius = data_radius 
        else:
            # normalize using the same mean and radius
            label_patches, centroid, furthest_distance = pc_utils.normalize_point_cloud(
                label_patches)
            input_patches = (input_patches - centroid) / furthest_distance
            radius = np.ones([B, 1], dtype=np.float32)


        input_patches = torch.from_numpy(input_patches).transpose(2, 1).squeeze(0).float().clamp(-1,1)
        label_patches = torch.from_numpy(label_patches).transpose(2, 1).squeeze(0).float()
        return input_patches, label_patches, data_radius




if __name__ == "__main__":
    dataset = H5Dataset(
        "../dataset/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5",
        num_shape_point=256, num_patch_point=256, batch_size=1)
    dataset.scales = [2]
    dataloader = data.DataLoader(dataset, batch_size=16, pin_memory=True, shuffle=True)
    for i, example in enumerate(dataloader):
        input_pc, label_pc, scales = example
        ratio = ratio.item()
        input_pc = input_pc[0].transpose(2, 1)
        label_pc = label_pc[0].transpose(2, 1)
        input_pc = input_pc.squeeze(1)
