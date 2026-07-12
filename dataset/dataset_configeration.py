import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
import random
import torch


TRAIN_PTH = "D:/datasets/Image Restoration dataset/train"
TEST_PATH = "D:/datasets/Image Restoration dataset/test"
#--------------------------------------------------------------------------
ZOOM_SIZE = 300
IMAGE_SIZE = 256
instruction_batch_size = 16
degradation_batch_size = 16
batch_size = 8
num_workers = 2
K = 4
tasks = ['LLIE', 'haze', 'noise', 'rain', 'UWIR', 'SR', 'blur']

def get_class_idx(class_name):
    task_idx = tasks.index(class_name)
    return task_idx

def get_Points():
    file_path = os.path.join(".", "data", "7_points.pt")
    loaded_points = torch.load(file_path)
    return loaded_points

def get_class_points(points, labels):
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    label_indices = [task_to_idx[label] for label in labels]
    label_indices = torch.tensor(label_indices)
    selected_points = points[label_indices]
    
    return selected_points, label_indices
#-----------------------------------------------------------------
tfms = transforms.Compose([       # Tensor
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
test_tfms = transforms.Compose([       # Tensor
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
weak_tfms = transforms.Compose([
    transforms.Resize((ZOOM_SIZE,ZOOM_SIZE)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(10, expand=False),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(brightness=.5, contrast=.1, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
#-----------------------------------------------------------------
def get_image_files(folder_path):
    # Define the allowed image file extensions
    allowed_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]  # Add more if needed

    image_files = []
    for ext in allowed_extensions:
        # Use glob to find files with allowed extensions in the folder path
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_files

def RandomSameCrop(img1, img2, scale=(0.8, 1.0)):
    size=(IMAGE_SIZE, IMAGE_SIZE)
    i, j, h, w = transforms.RandomResizedCrop.get_params(img1, scale=scale, ratio=(1.0, 1.0))
    # Apply the same crop to both images
    img1 = transforms.functional.resized_crop(img1, i, j, h, w, size)
    img2 = transforms.functional.resized_crop(img2, i, j, h, w, size)

    return img1, img2

def apply_transforms(img0, img1):

    img0 = TF.resize(img0, (IMAGE_SIZE, IMAGE_SIZE))
    img1 = TF.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
    
    if random.random() > 0.5:
        img0 = TF.vflip(img0)
        img1 = TF.vflip(img1)
    
    if random.random() > 0.5:
        img0 = TF.hflip(img0)
        img1 = TF.hflip(img1)

    return img0, img1
#-----------------------------------------------------------------