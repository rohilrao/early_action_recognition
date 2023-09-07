import cv2
import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets

dataset_dir = "/home/s6roraoo/smth-smth/videos/"
sample_proportion = 0.2
label_names = sorted(os.listdir(dataset_dir))
video_paths = []
labels = []


for label_id, label_name in enumerate(label_names):
    print(label_id, label_name)
    
    label_dir = os.path.join(dataset_dir, label_name)
    video_files = os.listdir(label_dir)

    # Calculate the number of videos to sample for this class
    num_videos_to_sample = int(len(video_files) * sample_proportion)

    # Randomly sample 'num_videos_to_sample' videos from this class
    sampled_videos = random.sample(video_files, num_videos_to_sample)

    for video_file in sampled_videos:
        video_path = os.path.join(label_dir, video_file)

        # Check if the video has more than 15 frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 45:  # Only include videos with 15 or fewer frames
            video_paths.append(video_path)
            labels.append(label_name)
            
    print(len(video_paths))

data_dict = dict()
data_dict['video_paths'] = video_paths
data_dict['labels'] = labels

import pickle

with open('smth_sample_0.2.pkl', 'wb') as file:
    # Serialize and save the list to the file
    pickle.dump(data_dict, file)