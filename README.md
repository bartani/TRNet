# if you see this message, It means we are uploading our source code and checkpoints. Please wait....
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

















## Contacts
For any inquiries contact Ako Bartani: <a href="mailto:a.bartani@uok.ac.ir">a.bartani [at] uok.ac.ir</a>
