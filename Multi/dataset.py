import os
import numpy as np

feature_path = '../Video/data/video'
audio_path = '../Audio/data/wav'

import torch
from torch.utils.data import Dataset, DataLoader

class fusionDataset(Dataset):
    def __init__(self, feature_path, audio_path, mode ='train'):
        self.feature_path = feature_path
        if mode == 'train':
            self.audio_path = np.load(os.path(audio_path,'X_train.npy'))
        else:
            self.audio_path = np.load(os.path(audio_path,'X_test.npy'))
        self.data_list = os.listdir(feature_path)

    def __len__(self):
        return audio_path.shape[0]

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        audio_data = self.audio_path[idx]
        video_data = np.load(os.path.join(self.feature_path,data_name))
        label = int(data_name.split('-')[2])
        return audio_data, video_data, label



custom_dataset = fusionDataset(feature_path, audio_path)

# DataLoader를 사용하여 데이터 로딩
batch_size = 32
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# DataLoader를 통해 데이터를 미니배치로 가져올 수 있음
for batch in dataloader:
    images, labels = batch