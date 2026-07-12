import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import Dataset, DataLoader
import dataset_configeration as config
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image


class train_dataset(Dataset):
    def __init__(self, imgs):
        self.files = imgs
        self.points = config.get_Points()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        image_path = Path(img_path)
        image_class = image_path.parent.name
        img_ = np.array(Image.open(img_path).convert('RGB'))

        w = img_.shape[1]
        w = int(w/2)
        x = img_[:, :w, :]
        y = img_[:, w:, :]

        x = Image.fromarray(np.uint8(x))
        y = Image.fromarray(np.uint8(y))

        # x, y = config.RandomSameCrop(x, y)
        x, y = config.apply_transforms(x,y)
        
        x = config.tfms(x)
        y = config.tfms(y)

        delta = self.points[config.get_class_idx(image_class)]
        
        return x, y, delta

def train_loader():
    myds = train_dataset(config.get_image_files(config.TRAIN_PTH+"/*"))
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader



if __name__ == "__main__":
    loader = train_loader()
    
    x, y, delta = next(iter(loader))
    print(x.shape)
    print(y.shape)
    print(delta.shape)


    # concat_cover = torch.cat((x*.5+.5, y*.5+.5), 2)

    # save_image(concat_cover, "outcomes/datasample/data.png")

    

    B, _, _, _ = x.shape
    for b in range(B):
        save_image(x[b,...]*.5+.5, f"outcomes/datasample/data-{b}.png")