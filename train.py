import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

import config
from dataset.mydataset import train_loader
from utility import init_Generator, save_model
from loss import L1, Perceptual
from tqdm import tqdm
import torch
from torchvision.utils import save_image





def train(model, opt, scr, loader, criterion, lp, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (x, y, delta) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)   
        delta = delta.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = model(x, delta)

            loss = criterion(y, fake)*config.L1_LAMBDA + lp(y, fake)

        opt.zero_grad()
        scr.scale(loss).backward()
        scr.step(opt)
        scr.update()

        loop.set_postfix(
            epoch=epoch,
            loss=loss.item(),
        )
    
    return model, opt, scr


def main():
    loader = train_loader()
    #---------------------------------------------------------------
    model, opt, scr = init_Generator()
    criterion = L1(config.DEVICE)
    Lp = Perceptual(config.DEVICE)
    #---------------------------------------------------------------
    for epoch in range(config.NUM_EPOCHS):
        model, opt, scr = train(model, opt, scr, loader, criterion, Lp, epoch)
        save_model(model, opt, config.GEN_checkpoints)


def test(): 
    loader = train_loader()
    
    # token = Tokenizer(config.DEVICE)
    # txt_enc,_,_ = init_txt_Model()
    # img_enc,_,_ = init_img_Model()
    model, _, _ = init_Generator()
    # txt_enc.eval()
    # img_enc.eval()
    model.eval()
    #--------------
    loop = tqdm(loader, leave=True)
    for idx, (x, y, delta) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        delta = delta.to(config.DEVICE)

        with torch.no_grad():
            fake = model(x, delta)
            concat_cover = torch.cat((x*.5+.5, y*.5+.5, fake*.5+.5), 2)
            save_image(concat_cover, f"outcomes/70/gen_{idx}.png")


if __name__ == "__main__":
    test()