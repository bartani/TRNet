
# TRNet (Tell and Restore Network)
## Contrastive Anchor-Based Task Representation for Instruction-Guided All-in-One Image Restoratio
### Authors: Ako Bartani, Omed Sedeeq Ahmad, Mohammed Shamsaddin Qadir, Karwan Ahmed Abdullah, and Fardin Akhlaghian Tab

![Alt text](img/model.png)

## Requirements
```
-PyTorch (CUDA-enabled for GPU training)
-torchvision
-numpy, scipy, pillow
-opencv-python
-tqdm, matplotlib
```
## Dataset preparation
We used seven degraded image dataset: deraining, dehazing, denoising, low-light image enhancement, underwater image restoration, deblurring, and image super-resolution. The both train and test image dataset are available at [Download Link](https://www.kaggle.com/datasets/akobartani/all-in-one-image-restoration). 

Moreover, you can find human instructions (train and test) and frozen task representation in the 'data' folder.
```
degraded dataset/
├── train/
│   ├── blur/
│   └── haze/
│   └── LLIE/
│   └── noise/
│   └── rain/
│   └── SR/
│   └── UWIR/
└── test/
│   ├── blur/
│   └── haze/
│   └── LLIE/
│   └── noise/
│   └── rain/
│   └── SR/
│   └── UWIR/
```
## Train Model
If you want to train the model using your dataset, we recommend that retrain the both modality encoder and task encoder model on new data. To this end, run the "degradation.py" and "instruction.py" using new data. Next, run the "train.py" on new dataset and ensure that in the config.py:
```
LOAD_checkpoints_Image_Encoder = False
TEXT_MODEL_checkpoints = False
GEN_checkpoints = False
```

Otherwise, you can use pre-trained checkpoints on defined dataset. To this end, you can download our checkpoints from:
- Generator checkpoints: [download link](https://drive.google.com/file/d/1-hzF56QsA9qD8ltUeAZiNLTdXKJ6wpNN/view?usp=sharing)
- Image encoder checkpoints: [download link](https://drive.google.com/file/d/145uC2VZG0qn7pHtTjqQj3TQeIQ8gEPWO/view?usp=sharing)
- Text encoder checkpoints: [download link](https://drive.google.com/file/d/184QG0ONHmuz9601kOZmlU-Xg6hIJjs7k/view?usp=sharing)

Please put downloaded checkpoints files at the "checkpoints/" and ensure that in the config.py: 
```
LOAD_checkpoints_modality_Encoder = True
LOAD_checkpoints_TEXT_Encoder = True
GENERATOR_LOAD_checkpoints = True
```
Also, you can change checkpoints path in the config.py as,
```
IMAGE_ENCODER_checkpoints = f"checkpoints/img_model.pth.tar"
TEXT_MODEL_checkpoints = f"checkpoints/txt_multilingualBERT.pth.tar" 
GEN_checkpoints = "checkpoints/gen-7D.pth.tar"
```

Also, you can change training dataset path in the dataset/dataset_configeration.py as,
```
TRAIN_PTH = "...datasets/Image Restoration dataset/train"
TEST_PATH = "...datasets/Image Restoration dataset/test"
```

## Test Model
To test model on 7 tasks:

1: ensure that the model checkpoints are available at the "checkpoints/" folder.

- Generator checkpoints: [download link](https://drive.google.com/file/d/1-hzF56QsA9qD8ltUeAZiNLTdXKJ6wpNN/view?usp=sharing)
- Image encoder checkpoints: [download link](https://drive.google.com/file/d/145uC2VZG0qn7pHtTjqQj3TQeIQ8gEPWO/view?usp=sharing)
- Text encoder checkpoints: [download link](https://drive.google.com/file/d/184QG0ONHmuz9601kOZmlU-Xg6hIJjs7k/view?usp=sharing)

2: at the config.py:
```
LOAD_checkpoints_Image_Encoder = True
TEXT_MODEL_checkpoints = True
GEN_checkpoints = True
```

3: Please set the degraded image path, instruction, and save path in the test.py
```
save_path = "save path result"
Image_path = "your degraded image path"
your_instruction = "write your instruction"
```

4: run the "test.py".

## Citation

The citation information will be included after the published.














## Contacts
For any inquiries contact Ako Bartani: <a href="mailto:a.bartani@uok.ac.ir">a.bartani [at] uok.ac.ir</a>
