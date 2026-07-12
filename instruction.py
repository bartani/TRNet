import sys
sys.path.append('models')
sys.path.append('dataset')

from dataset.instruction_dataset import train_loader
import config
from tqdm import tqdm
import torch
from loss import CosineTripletLoss
from utility import init_txt_Model, save_model
from models.task_encoder import multilingualBERT_Tokenizer



def Text_train(model, opt, scr, token, criterion, loader, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (d) in enumerate(loop):
        
        pos_text = d['pos_text']
        neg_text = d['neg_text']
        
        anc = d['anc'].to(config.DEVICE)

        p = token(pos_text)
        n = token(neg_text)

        with torch.cuda.amp.autocast():
            pos = model(p)
            neg = model(n)
            
            loss = criterion(anc, pos, neg)

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
    model, opt, scr = init_txt_Model()
    token = multilingualBERT_Tokenizer(config.DEVICE) 
    #-----------------------------------------------------------------
    criterion = CosineTripletLoss()
    #-----------------------------------------------------------------

    for epoch in range(50):
        model, opt, scr = Text_train(model, opt, scr, token, criterion, trn_loader, epoch)        
        save_model(model, opt, config.TEXT_MODEL_checkpoints)

if __name__ == "__main__":
    main()