import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

from utility import init_img_Model, init_txt_Model, init_Generator
from models.task_encoder import multilingualBERT_Tokenizer
import config
from dataset.test_data import praparing_image
from dataset.dataset_configeration import get_Points, tasks
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

def enhance_image(img_path, instruction, save_path):
    
    Points = get_Points().to(config.DEVICE)
    token = multilingualBERT_Tokenizer(config.DEVICE)
    txt_enc,_,_ = init_txt_Model()
    img_enc,_,_ = init_img_Model()
    model, _, _ = init_Generator()
    txt_enc.eval()
    img_enc.eval()
    model.eval()

    x = praparing_image(img_path).to(config.DEVICE)
    
    with torch.no_grad():
        img_features = img_enc(x)
        txt_features = txt_enc(token([instruction]))

    
    dists = torch.norm(img_features - Points, dim=1)
    sim = 1- F.cosine_similarity(txt_features, Points, dim=1)
    final = dists + sim
    target_task = final.argmin()


    delta = Points[target_task].unsqueeze(0) 
    with torch.no_grad():
        enhanced = model(x, delta)
        save_image(enhanced*.5+.5, save_path)
    print("Based on your input image and instruction, TRNet applaied", tasks[target_task], "restoration task on the input image")




if __name__ == "__main__":
    # your_instruction = "Can you help clean up this picture? The noise is ruining the details." # task noise
    # your_instruction = "Cette photo est couverte de gouttes de pluie. Pouvez-vous les supprimer pour que ce soit clair" # rain noise
    # your_instruction = "Restore the brilliant orange glow of the tube sponges nestled among the reef rocks." # task WUIR

    
    your_instruction = "please enhance my photo" # vauge
    
    Image_path = "data/test images/single degradation/data-2.png"
    save_path = "outcomes/gen/data-2.png"


    enhance_image(Image_path, your_instruction, save_path)