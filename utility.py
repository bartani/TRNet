import torch
import config
import torch.optim as optim
from models.task_encoder import Text_Encoder, Image_Encoder
from models.generator import Generator
from torchvision.utils import save_image
import torch.nn.functional as F
from dataset.test_data import praparing_image



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def save_model(model, opt, path):
    save_checkpoint(model, opt, filename=path)

def init_txt_Model():
    model = Text_Encoder(in_dim=768).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_checkpoints_TEXT_Encoder:
        load_checkpoint(config.TEXT_MODEL_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt, scr

def init_img_Model():
    model = Image_Encoder(in_channels=3, features=64).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_checkpoints_Image_Encoder:
        load_checkpoint(config.IMAGE_ENCODER_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt, scr

#================================================================================================
def init_Generator():
    model = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.GENERATOR_LOAD_checkpoints:
        load_checkpoint(config.GEN_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt, scr


def save_some_examples(model, img_enc, txt_enc, token, Points, loader, epoch, folder):
    
    x, y, instruction, image_class, delta = next(iter(loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    delta = delta.to(config.DEVICE) #get_tasks_representation(img_enc, txt_enc, token, x, instruction, Points)

    model.eval()
    with torch.no_grad():
        fake = model(x, delta)
        concat_cover = torch.cat((x*.5+.5, y*.5+.5, fake*.5+.5), 2)
        save_image(concat_cover, folder + f"gen_{epoch}.png")
      
    model.train()


def enhance_image(img_path, instruction, deg_enc, inst_enc, token, gen, Points, save_path):

    Points = Points.to(config.DEVICE)
    x = praparing_image(img_path).to(config.DEVICE)
    x = x.repeat(8, 1, 1, 1)

    with torch.no_grad():
        img_features = deg_enc(x)
        txt_features = inst_enc(token([instruction]))
    
    sim = F.cosine_similarity(txt_features, Points, dim=1)
    target_task = sim.argmax()

    delta = Points[target_task].unsqueeze(0) 
    delta = delta.repeat(8, 1)
    with torch.no_grad():
        enhanced = gen(x, delta)
        save_image(enhanced*.5+.5, save_path)


    return delta