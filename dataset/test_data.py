

from PIL import Image
import dataset_configeration as config

def praparing_image(path):
    x = Image.open(path).convert('RGB')
    return config.tfms(x).unsqueeze(0)