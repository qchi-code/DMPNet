# DMPNet
🎉Our manuscript has been submitted.

## 🚩Abstract
Few-shot medical image segmentation (FSMIS) aims to segment unseen anatomical structures using only ...

## 💡Motivation

## 🔍Overview of DMPNet

## 🗝️Quick start
### 🔖1. Dependencies
Please install the following dependencies:

```python
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.24.4
opencv_python==4.11.0.86
Pillow>=8.1.1
sacred==0.8.7
scikit_learn==1.3.2
scikit-image==0.18.3
SimpleITK==2.5.2
torch==2.5
torchvision==0.19.1
matplotlib==3.7.5
scipy==1.16.0
```

### 📋2. Datasets and Pre-processing
1. Download Datasets:
+ Abdomen MRI：[Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)
+ Abdomen CT：[Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
+ <font style="color:rgb(31, 35, 40);">Cardiac LGE and b-SSFP</font>：[Multi-sequence Cardiac MRI Segmentation dataset](https://zmiclab.github.io/zxh/0/mscmrseg19/index.html)
+ <font style="color:rgb(31, 35, 40);">Prostate UCLH and NCI</font>：[Cross-institution Male Pelvic Structures](https://zenodo.org/records/7013610)
2. Data Pre-processing:
+ <font style="color:rgb(31, 35, 40);">Pre-processing is performed according to </font>[<font style="color:rgb(9, 105, 218);">Ouyang et al.</font>](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535)<font style="color:rgb(31, 35, 40);"> and we follow the procedure on their GitHub repository.</font>


### 📍Download ResNet Pre-trained Weights
| resnet50-imagenet | [https://download.pytorch.org/models/resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) |
| --- | --- |
| resnet50-coco | [https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth) |
| resnet101-imagenet | [https://download.pytorch.org/models/resnet101-63fe2227.pth](https://download.pytorch.org/models/resnet101-63fe2227.pth) |
| resnet101-coco | [https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) |


### 🔥 Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py`
2. Download pre-trained ResNet-101 weights and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./scripts/train_<direction>.sh`, for example: `./scripts/train_CHAOST2_setting1.sh`
4. The trained model weights are available at the following link for download and evaluation: [Google Drive](https://drive.google.com/file/d/1qnlC9Z158WsUY47YTZiAwiJr8QseHXJt/view?usp=drive_link). Please note that you need to manually download and extract the files.


## 🌹Acknowledgements
<font style="color:rgb(31, 35, 40);">Our code is built upon the works of </font>[<font style="color:rgb(9, 105, 218);">SSL-ALPNet</font>](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)<font style="color:rgb(31, 35, 40);">, </font>[<font style="color:rgb(9, 105, 218);">ADNet</font>](https://github.com/sha168/ADNet)<font style="color:rgb(31, 35, 40);"> and </font>[<font style="color:rgb(9, 105, 218);">QNet</font>](https://github.com/ZJLAB-AMMI/Q-Net)<font style="color:rgb(31, 35, 40);">, we appreciate the authors for their excellent contributions!</font>
