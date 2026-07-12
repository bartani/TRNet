import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("dataset")

from dataset.dataset_configeration import get_Points, TEST_PATH, tasks
import torchvision.datasets as datasets
from dataset.degradation_dataset import test_loader
from utility import init_img_Model
from tqdm import tqdm
import config
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def extract_image_embedding():
    points = get_Points()
    loader = test_loader()
    model, _, _ = init_img_Model()
    model.eval()
    #-----------------------------------------------------------------
    embeddings = []
    labels = []
    #-----------------------------------------------------------------
    loop = tqdm(loader, leave=True)
    for idx, (x, lbl) in enumerate(loop):
        
        x = x.to(config.DEVICE)

        # features = x.view(x.size(0), -1)
        # embeddings.append(features.cpu())
        # labels.append(lbl.cpu())

        with torch.no_grad():
            features = model(x)
            embeddings.append(features.cpu())
            labels.append(lbl.cpu())
    
    
    embeddings.append(points)
    for i in range(len(points)):
        labels.append(torch.tensor([i]))    
    
    
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings, labels

def tSNE_Plot():
    points = get_Points()
    dataset = datasets.ImageFolder(TEST_PATH)
    
    embeddings, labels = extract_image_embedding()


    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d  = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(dataset.classes))

    for class_idx, class_name in enumerate(dataset.classes):
        indices = labels == class_idx
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    label=class_name, alpha=0.6, s=30, color=palette[class_idx])

    
    colors = plt.cm.tab10.colors  # Get 10 distinct colors
    for i, (x, y) in enumerate(embeddings_2d[len(embeddings_2d)-len(points):]):
        plt.scatter(x, y, color='black', marker='X', s=100, edgecolors='black')
        plt.text(x + 1.5, y, tasks[i], fontsize=12, weight='bold', color='black')

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tSNE_Plot()