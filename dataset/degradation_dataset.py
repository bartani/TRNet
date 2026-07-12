import dataset_configeration as config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import random
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path




TRAIN_IMG=datasets.ImageFolder(config.TRAIN_PTH)
TEST_IMG=datasets.ImageFolder(config.TEST_PATH)
class train_Dataset(Dataset):
    
    def __init__(self, ):
        self.imgs = TRAIN_IMG.imgs
        self.points = config.get_Points()
        self.class_name = TRAIN_IMG.classes

        
    def __getitem__(self, index):
        # Select anchor image and its label from dataset
        positive, pos_lbl = self.imgs[index]
        system_class = self.class_name[pos_lbl]
        anc = self.points[config.get_class_idx(system_class),:]
        
        # Randomly select a negative example
        negative, neg_lbl = random.choice([(img, lbl) for img, lbl in self.imgs if lbl != pos_lbl])
        
        # read the images (anchor, positive and negative images)
        
        pos_img = np.array(Image.open(positive).convert('RGB'))
        
        w_pos = pos_img.shape[1]
        w_pos = int(w_pos/2)
        pos_img = pos_img[:, :w_pos, :]
        
        pos_img = Image.fromarray(np.uint8(pos_img))

        neg_img = np.array(Image.open(negative).convert('RGB'))
        w_neg = neg_img.shape[1]
        w_neg = int(w_neg/2)
        neg_img = neg_img[:, :w_neg, :]

        neg_img = Image.fromarray(np.uint8(neg_img))
        
        # perform any required transformation (if any)
        pos_img = config.weak_tfms(pos_img)
        neg_img = config.weak_tfms(neg_img)
        
        return anc, pos_img, neg_img
    
    def __len__(self):
        return len(self.imgs)
    

class test_Dataset(Dataset):
    def __init__(self, imgs):
        self.files = imgs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        image_path = Path(img_path)
        image_class = image_path.parent.name
        lbl = config.get_class_idx(image_class)
        img_ = np.array(Image.open(img_path).convert('RGB'))
        w = img_.shape[1]
        w = int(w/2)
        x = img_[:, :w, :]

        x = Image.fromarray(np.uint8(x))
        
        x = config.test_tfms(x)
        
        return x, lbl
    

def train_loader():
    _ds = train_Dataset()    
    _dl = DataLoader(_ds, batch_size=config.degradation_batch_size, shuffle=True, num_workers=config.num_workers)
    return _dl

def test_loader():
    _ds = test_Dataset(config.get_image_files(config.TEST_PATH+"/*"))    
    _dl = DataLoader(_ds, batch_size=1, shuffle=True, num_workers=config.num_workers)
    return _dl
    
if __name__ == "__main__":
    
    loader = train_loader()
    
    anc, pos, neg = next(iter(loader))
    print(pos.shape, neg.shape)
    print(anc.shape)

    concat_cover = torch.cat((pos*.5+.5, neg*.5+.5), 2)

    save_image(concat_cover, "outcomes/datasample/sample_pos_neg.png")