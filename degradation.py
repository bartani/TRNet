import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

import config
from tqdm import tqdm
import torch
from dataset.degradation_dataset import train_loader
from utility import init_img_Model, save_model
from loss import TripletLoss


def Image_train(model, opt, scr, criterion, loader, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (anc,pos,neg) in enumerate(loop):
        
        anc, pos, neg = anc.to(config.DEVICE), pos.to(config.DEVICE), neg.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            pos_ = model(pos)
            neg_ = model(neg)

            loss = criterion(anc, pos_, neg_)

        opt.zero_grad()
        scr.scale(loss).backward()
        scr.step(opt)
        scr.update()


        loop.set_postfix(
            LOSS=loss.item(),
            epoch=epoch,
        )
    return model, opt, scr

def main():
    trn_loader = train_loader()
    #-----------------------------------------------------------------
    model, opt, scr = init_img_Model()
    #-----------------------------------------------------------------
    criterion = TripletLoss()
    #-----------------------------------------------------------------

    for epoch in range(50):
        model, opt, scr = Image_train(model, opt, scr, criterion, trn_loader, epoch)        
        save_model(model, opt, config.IMAGE_ENCODER_checkpoints)

if __name__ == "__main__":
    main()