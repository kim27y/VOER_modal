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

# Video Feature Extraction
## 1. Download slowfast bencmark
- Download [Here](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md)
- It is better to download slowfast bencmark 'SLOWFAST_8x8_R50.pkl'
## 2. Set SLOWFAST_8x8_R50.yaml file
- Go to ```SLOWFAST_8x8_R50.yaml``` file
  ```bash
  cd Video/slowfast_feature_extractor
  cd config
  ```
- You should check input, output path

# Inference
**CHECKPOINT**: The checkpoints of X3D, MVitV2 and ActionFormer from our training process can be found [here]()
### Step 1: Feature extraction
1. Feature extraction:
```bash
# For X3D model:
cd /PySlowFast-X3D
python tools/extract_feature.py --cfg configs/AICity2023/X3D-L_extract_feature.yaml

# For MViTv2 model:
cd /PySlowFast-X3D
python tools/extract_feature.py --cfg configs/AICity2023/MViTv2_extract_feature.yaml
```

**Note**: Please modify the variables in the file config .yaml as follows:
- DATA.PATH_TO_DATA_DIR: path to dataset input, e.g., /data/SetB.
- DATA.PATH_EXTRACT: path to feature output, e.g., /data/featureB.
- WEIGHT: path to checkpoints' model.

2. Concatenate views/models (optional):
```bash
cd /data_processing/concat_view_features.py --feature_dir /path/to/feature \
                                            --output_dir /path/to/feature_output

cd /data_processing/concat_model_features.py --feature_dir_mvit /path/to/feature_mvit \
                                            --feature_dir_x3d /path/to/feature_x3d \
                                            --output_dir /path/to/feature_output
```
### Step 2: Action localization
1. Create data json format
```bash
cd /data_processing
python create_data_json_action_former_without_gt.py --dataset_dir /path/to/SetB \
                                                    --feature_dir /path/to/feature_B \
                                                    --output_dir /path/to/json_format
```

2. Action localization
```bash
cd /action_localization
python ./inference_all.py ./config/AIC_2023_X3D_MViTv2.yaml \
                            /path/to/checkpoint \
                            --num_folds 5 \
                            --output_dir /path/to/raw_prediction
```
**Note**: Please modify the variables in the file config .yaml as follows:
- dataset.json_file: path to data json
- dataset.input_dim: the clip feature dimension

3. Create submission
```bash
cd /data_processing
python convert_to_submission.py --dataset_path /path/to/SetB \
                                --prediction_path /path/to/raw_prediction
```
# Results

# Citation
If you find our work useful, please use the following BibTeX entry for citation.

# Acknowledgement
Codebases: [PySlowFast](https://github.com/facebookresearch/SlowFast) and [ActionFormer](https://github.com/happyharrycn/actionformer_release)


# Contact
Huy Duong Le (duonglh9@viettel.com.vn / huyduong7101@gmail.com)

Manh Tung Tran (tungtm6@viettel.com.vn)

Minh Quan Vu (quanvm4@viettel.com.vn)