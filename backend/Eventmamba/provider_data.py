import os
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset,DataLoader

def load_h5_mark(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    mark = f['mark'][:]
    return (data, label,mark)

def load_h5(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_and_resample(h5_filename, sample_size=1024):
    print(f"Loading {h5_filename}")
    with h5py.File(h5_filename, 'r') as f:
        data = []
        labels = []
        for group_name in f:
            group = f[group_name]
            x = group['x'][:]
            y = group['y'][:]
            t = group['t'][:]
            current_sample_size = len(x)
            if current_sample_size < sample_size:
                repeat_factor = np.ceil(sample_size / current_sample_size).astype(int)
                x = np.tile(x, repeat_factor)[:sample_size]
                y = np.tile(y, repeat_factor)[:sample_size]
                t = np.tile(t, repeat_factor)[:sample_size]
            else:
                indices = np.random.choice(current_sample_size, sample_size, replace=False)
                indices = np.sort(indices)
                x = x[indices]
                y = y[indices]
                t = t[indices]
            if current_sample_size >= 1024:
                t = t * 0.1
                sample_data = np.stack((t,x,y), axis=-1)
                data.append(sample_data)
                sample_label = group.attrs['label']
                labels.append(sample_label)
    return (data, labels)

def load_h5_and_resample_INI30(h5_filename, sample_size=1024):
    print(f"Loading {h5_filename}")
    with h5py.File(h5_filename, 'r') as f:
        data = []
        labels = []
        for ID in f:
            sample = f[ID]
            for group_name in sample:
                group = sample[group_name]
                x = group['x'][:]
                y = group['y'][:]
                t = group['t'][:]
                current_sample_size = len(x)
                if current_sample_size < sample_size:
                    repeat_factor = np.ceil(sample_size / current_sample_size).astype(int)
                    x = np.tile(x, repeat_factor)[:sample_size]
                    y = np.tile(y, repeat_factor)[:sample_size]
                    t = np.tile(t, repeat_factor)[:sample_size]
                else:
                    indices = np.random.choice(current_sample_size, sample_size, replace=False)
                    indices = np.sort(indices)
                    x = x[indices]
                    y = y[indices]
                    t = t[indices]
                if current_sample_size >= 1024:
                    # t = t * 0.1                             #放缩时间
                    sample_data = np.stack((t,x,y), axis=-1)
                    data.append(sample_data)
                    sample_label = group.attrs['label']
                    labels.append(sample_label)
    return (data, labels)



class EyeTrackingChunkDataset(Dataset):
    def __init__(self, h5_filename, chunk_size=64, sample_size=1024):
        self.h5_filename = h5_filename
        self.chunk_size = chunk_size
        self.sample_size = sample_size
        
        # 加载并预处理数据
        self.chunks = self._load_and_process_data()

    def _load_and_process_data(self):
        data_chunks = []
        
        with h5py.File(self.h5_filename, 'r') as f:
            for ID in f:
                sample = f[ID]
                try:
                    frame_keys = sorted(sample.keys(), key=lambda x: int(x))
                except ValueError:
                    frame_keys = sorted(sample.keys(), key=lambda x: int(x.split('_')[-1]))


                video_frames_list = []
                video_labels_list = []

                for group_name in frame_keys:
                    group = sample[group_name]
                    
                    x = group['x'][:]
                    y = group['y'][:]
                    t = group['t'][:]
                    
                    current_sample_size = len(x)
                    
                    if current_sample_size < 1024:
                        continue 

                    if current_sample_size < self.sample_size:
                        repeat_factor = np.ceil(self.sample_size / current_sample_size).astype(int)
                        x = np.tile(x, repeat_factor)[:self.sample_size]
                        y = np.tile(y, repeat_factor)[:self.sample_size]
                        t = np.tile(t, repeat_factor)[:self.sample_size]
                    else:
                        indices = np.random.choice(current_sample_size, self.sample_size, replace=False)
                        indices = np.sort(indices)
                        x = x[indices]
                        y = y[indices]
                        t = t[indices] * 0.1   # 控制t的缩放
                    frame_data = np.stack((t, x, y), axis=-1)
                    
                    if 'label' in group.attrs:
                        frame_label = group.attrs['label']
                    else:
                        print(f"Warning: No label found for {ID}/{group_name}, skipping.")
                        continue

                    video_frames_list.append(frame_data)
                    video_labels_list.append(frame_label)

                total_frames = len(video_frames_list)
                
                if total_frames >= self.chunk_size:
                    
                    video_tensor = np.stack(video_frames_list, axis=0)
                    label_tensor = np.array(video_labels_list)
                    
                    for i in range(0, total_frames, self.chunk_size):
                        
                        if i + self.chunk_size <= total_frames:
                            chunk_data = video_tensor[i : i + self.chunk_size]
                            chunk_label = label_tensor[i : i + self.chunk_size]
                            
                            data_chunks.append({
                                'data': torch.from_numpy(chunk_data).float(), # Shape: [64, 1024, 3]
                                'label': torch.from_numpy(chunk_label).float() # Shape: [64]
                            })
                            
        print(f"数据加载完成。共生成 {len(data_chunks)} 个序列片段 (Shape: [{self.chunk_size}, {self.sample_size}, 3])")
        return data_chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        item = self.chunks[idx]
        return item['data'], item['label']