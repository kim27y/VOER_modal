# -*- coding: utf-8 -*-

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

from utils import TemporalSegmentSubsample, prepare, PackPathway, time_to_second
from tqdm import tqdm
import os
import pandas as pd
import math
import torch
import glob
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Feature_extract_hyperparaters')
parser.add_argument('--video_path',
                    type=str,
                    default='../../datas/',
                    help='data sort, train or test')
parser.add_argument('--num_segments',
                    type=int,
                    default=4,
                    help='window size segment number')
parser.add_argument('--fps',
                    type=int,
                    default=16,
                    help='frames_per_segments')
parser.add_argument('--feature_size',
                    type=int,
                    default=224,
                    help='data sort, train or test')


def make_transform(num_segments, frames_per_segment, size):
    mean = [0.45, 0.45, 0.45],
    std = [0.225, 0.225, 0.225]
    slowfast_alpha = 4
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                TemporalSegmentSubsample(num_segments=num_segments, 
                                            frames_per_segment=frames_per_segment,
                                            test_mode=False),
                    # Lambda(lambda x: x/255.0),
                    # NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=size
                    ),
                    PackPathway(slowfast_alpha=slowfast_alpha)
            ]
        ),
    )
    
    
    return transform


def feature_extractor(model,
                    video_path: str,
                    device: str,
                    transform: object) -> torch.Tensor:

    # Initialize an EncodedVideo helper class and load the video
    # Load the desired clip
    video = EncodedVideo.from_path(video_path)

    
    video_data = video.get_clip(start_sec=0, end_sec=math.inf)

    video_data = transform(video_data)
    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]
    """ Get Predictions"""
    # Pass the input clip through the model

    preds = model(inputs)
    
    return preds


if __name__ == '__main__':
    args = parser.parse_args()
    video_path = args.video_path
    num_segments = int(args.num_segments)
    frames_per_segment = int(args.fps)
    feature_size = args.feature_size

    # Choose the `slowfast_r101` model
    model = torch.hub.load('facebookresearch/pytorchvideo',
                        'slowfast_r101',
                        pretrained=True)
    device = "cuda"
    model = prepare(model,inplace=False,size=feature_size)

    model = model.eval()
    model = model.to(device)
    
    ################################################################################
    # train
    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    # num_frames = num_segments * frames_per_segment
    p = num_segments / frames_per_segment  # success probability
    #R, G, B
    fp = 0.01  # False probability 32, 48, 64
    trial = int((1/p)*math.log(1/fp))


    print("ㅎㅇㅎㅇ")

    Folder = os.listdir(video_path)
    for folder in Folder:
        folder_path = os.path.join(video_path, folder)
        for video in os.listdir(folder_path):
            input_path = os.path.join(folder_path,video)
            output_path = os.path.join('data','features2', folder)

            transform = make_transform(num_segments, frames_per_segment, size = feature_size)
            os.makedirs(output_path, exist_ok=True)
            output_name = video.split('.')[0] + ".npy"
            pred = feature_extractor(model=model,
                                            video_path=input_path,
                                            device=device,
                                            transform=transform)
            RGB_features = [pred.cpu().detach().numpy()]
            RGB_features = np.dstack(RGB_features)
            np.save(os.path.join(output_path, output_name),RGB_features)
            # for i in range(trial):
            #     output_name = video.split('.')[0] + f'_{i}' + ".npy"
            #     if os.path.exists(os.path.join(output_path, output_name)):
            #             continue
            #     pred = feature_extractor(model=model,
            #                                 video_path=input_path,
            #                                 device=device,
            #                                 transform=transform)
            #     RGB_features = [pred.cpu().detach().numpy()]
            #     RGB_features = np.dstack(RGB_features)
            #     np.save(os.path.join(output_path, output_name),RGB_features)
