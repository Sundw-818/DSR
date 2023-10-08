import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio

from lib.utils import *

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        sample_fraction = 1.0,
        radius = 25.0
    ):
        self.fraction = sample_fraction
        self.radius = radius
        self.data_dir = data_dir
        self.datafiles = get_instance_filenames(data_dir)
        self.num_sequences = len(self.datafiles)    
        
        #self.sample_size = int(self.fraction * len(self.dataset[0][0]))    

        # Load the whole dataset into CPU memory
        self.dataset = []
        print("loading dataset into memory")
        for idx, datafile in enumerate(self.datafiles):
            filename = os.path.join(self.data_dir, datafile)
            self.dataset.extend(unpack_sdf_samples_fraction(filename, self.fraction, self.radius, sequence_id=idx))

    def __len__(self): # total number of frames across all sequences
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]

class RawDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, extension='obj', d_in=3, with_normal=True, s=None,t=None):
        self.data_dir = data_dir
        self.extension = extension
        self.with_normal = with_normal
        self.datafolders = get_instance_folders(self.data_dir)
        self.num_sequences = len(self.datafolders)
        
        # Calc normalization scale and translation
        self.folder_list = [os.path.join(self.data_dir, folder) for folder in self.datafolders]
        
        self.frame_num_per_seq = []
        
        for folder_name in self.folder_list:
            objfiles = get_instance_filenames(folder_name, extension=extension)
            self.frame_num_per_seq.append(len(objfiles))
        
        if t and s:
            self.t = t
            self.s = s
        else:
            self.t, self.s = self.extract_mesh_radius(self.folder_list, extension=self.extension)
        
        # Load the whole dataset into CPU memory
        self.dataset = []
        print('loading dataset into memory...')
        
        for idx, folder_name in enumerate(self.folder_list):
            
            self.dataset.extend(unpack_point_samples(folder_name=folder_name, 
                                                     extension=self.extension,
                                                     with_normal=self.with_normal, 
                                                     translate=self.t,
                                                     scale=self.s, 
                                                     sequence_id=idx))
        print('data loaded!')
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def extract_mesh_radius(self, folder_list, extension='obj'):
        data = []
        for folder_name in folder_list:
            objfiles = get_instance_filenames(folder_name, extension=extension)
            objfiles.sort(key=lambda i: int(i.split('_')[-1][0:-(len(extension)+1)]))
            for file in tqdm(objfiles, desc=f'processing, {folder_name}'):
                raw_data = load_point_cloud_by_file_extension(os.path.join(folder_name, file), with_normal=False)
                data.append(raw_data)
        all_coord = np.concatenate(data, axis=0)
        
        # t = (all_coord[:,0].min() + all_coord[:,1].min() + all_coord[:,2].min()) / 3
        t = all_coord[:,0:3].min()
        d_x = all_coord[:,0].max() - all_coord[:,0].min()
        d_y = all_coord[:,1].max() - all_coord[:,1].min()
        d_z = all_coord[:,2].max() - all_coord[:,2].min()
        s = max(d_x, d_y, d_z) * 1.1
        return t, s
    def get_s_and_t(self):
        return self.s, self.t
