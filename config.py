
import torch
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#-----------------------------------------------------------------
LOAD_checkpoints_Image_Encoder = True
IMAGE_ENCODER_checkpoints = f"checkpoints/img_model.pth.tar"
#-----------------------------------------------------------------
TEXT_MODEL_checkpoints = f"checkpoints/txt_multilingualBERT.pth.tar" 
LOAD_checkpoints_TEXT_Encoder = True
#-----------------------------------------------------------------
GENERATOR_LOAD_checkpoints = True
GEN_checkpoints = "checkpoints/gen-7D.pth.tar"
#-----------------------------------------------------------------
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2000
L1_LAMBDA = 100
