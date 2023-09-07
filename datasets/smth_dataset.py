import cv2
import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from datasets.video_capture import VideoCapture
from config.base_config import Config
import pickle

class SomethingSomethingDataset(Dataset):
    def __init__(self, config: Config, split_type = 'train', img_transforms=None ):

        #dataset_dir, sample_proportion=1.0

        with open('/home/s6roraoo/xpool-main/datasets/smth_sample_0.2.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)
        
        self.video_paths = data_dict['video_paths']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        video_frames = VideoCapture.load_frames_from_video(video_path, percent=0.3)
        
        video_id = video_path.split('/')[-1].split('.')[0]

        ret = {
            'video_id': video_id,
            'video': video_frames[0],
            'text': label
        }

        return ret