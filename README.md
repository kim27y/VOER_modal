# [KCI] VAER_MODAL: Voice-Audio Emotion Recognition Multi Modal model
# Introduction
- We study 'Voice-Audio Emotion Recognition' for [RAVDESS](https://zenodo.org/record/1188976#.YFZuJ0j7SL8) dataset. So, we use Video and Audio data for do this, and We propose 'Multi-Modal model' for using two datasets that have different domains simultaneously.

- We propose **VAER_MODAL: Voice-Audio Emotion Recognition Multi Modal model** for solve the problem. This methodlogy's core is "Feature Extraction" and "Multi-Modal model"

# Installation
Please find installation instructions for PySlowFast in [here](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md)

1. Ubuntu 16.x/18.x (Only tested on these two systems)
2. Cuda 10.2
3. Python >= 3.7
4. [Pytorch](https://pytorch.org/)  >= 1.6
5. [PySlowFast](https://github.com/facebookresearch/SlowFast.git) >= 1.0
6. PyAv >= 8.x
7. Moviepy >= 1.0
8. OpenCV >= 4.x

It is recommended to use conda environment to install the dependencies.

You can create the conda environment with the command:

```
conda create -n "slowfast" python=3.7
```

Install Pytorch 1.6 and Torchvision 0.7 with conda or pip. (https://pytorch.org/get-started/locally/)

Install the following dependencies with pip:

```
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson av psutil opencv-python tensorboard moviepy cython
```

Install detectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Setup Pyslowfast:
```
git clone https://github.com/facebookresearch/slowfast
export PYTHONPATH=/path/to/slowfast:$PYTHONPATH
cd slowfast
python setup.py build develop
```
# Data Preparation
### Step 1: Download RAVDESS dataset (Audio+Video)
- Click to [RAVDESS HomePage](https://zenodo.org/record/1188976#.YFZuJ0j7SL8), Download RAVDESS data(Audio+Video), not Song version!
- If you download all **RAVDESS Dataset**, Dataset directory in the following way:
```
|---<path to dataset>
|   |---Actor_01
|   |   |---01-02-01-01-01-01-01.mp4
|   |   |---01-02-01-01-01-02-01.mp4
|   |   |---.
|   |---Actor_02
|   |   |---video_2
|   |   |   |---01-02-01-01-01-01-02.mp4
|   |   |   |---01-02-01-01-01-02-02.mp4
|   |   |   |---.
|   .   .  
|   .   .
|   .   .
|---|---Actor_24
|   |   |   |---01-02-01-01-01-01-24.mp4
|   |   |   |---01-02-01-01-01-02-24.mp4
|   |   |   |---.

```

### Step 2: Extract the Audio data from RAVDESS dataset
```bash
cd .
python Data_preprocess.py
```

And then, you can get audio('*.wav) data to path ('./Audio/data/audio')

## Data structure
```
workspace/
    ├── RAVDES_Data
    │   ├── Actor_01
    │   ├── ...
    ├── Audio
    │   ├── data
    │   |   ├──audio
    │   |   |  ├── Actor_01
    │   |   |  ├── ...
    ├── Video
    ├── Multi
```

# Video Model Inference
## 1. Download slowfast bencmark
- Download [Here](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md)
- It is better to download slowfast bencmark 'SLOWFAST_8x8_R50.pkl'
## 2. Set SLOWFAST_8x8_R50.yaml file
- Go to ```SLOWFAST_8x8_R50.yaml``` file
  ```bash
  cd Video/slowfast_feature_extractor
  cd config
  ```
- You should check input, output path and set that to video dataset input and output path and checkpoint path

## 3. Do feature Extractor
- To extract feature, execute the run_net.py as follow:
```
python run_net.py --cfg ./configs/<config_file>.yaml
```

## 4. Set feature dataset

- dataset python file is in ```workspace/Video/video_preprocessing.py```
- You can make dataset npy file to ```./Video/data/X_train.npy``` and ```./Video/data/Y_train.npy```

## 5. Do inference and check result


# Audio model inference

# Results

# Citation
You can citate my paper in "[한국중소기업학회](http://kasbs.or.kr/index.asp)"

# Acknowledgement
Codebases: [PySlowFast](https://github.com/facebookresearch/SlowFast) and [slowfast feature extractor](https://github.com/tridivb/slowfast_feature_extractor/tree/master)


# Contact
Jong Gu Kim (kim27y@naver.com)