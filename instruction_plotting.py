import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("dataset")

import torch
from dataset.dataset_configeration import get_Points, tasks, get_class_idx
from utility import init_txt_Model
import numpy as np
import pandas as pd
import config
from models.task_encoder import multilingualBERT_Tokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def extract_embedding(data):
    points = get_Points()
    model, _, _ = init_txt_Model()
    token = multilingualBERT_Tokenizer(config.DEVICE)
    model.eval()
    #-----------------------------------------------------------------
    embeddings = []
    labels = []
    #-----------------------------------------------------------------
    for i in range(len(data)):
        row = data.iloc[i]   
        text = row['text']
        lbl = torch.tensor([get_class_idx(row['class'])])

        
        # with torch.no_grad():
        #     t = token(text)
        #     t = t.view(t.shape[0], -1)
        #     embeddings.append(t.cpu())
        #     labels.append(lbl)


        with torch.no_grad():
            t = token(text)
            t = model(t)
            embeddings.append(t.cpu())
            labels.append(lbl)
    
    
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
    file_path = os.path.join(".", "data", "7-train_instruction_translated.csv")
    data = pd.read_csv(file_path)
    len_data = len(data)
    print(len_data)
    
    embeddings, labels = extract_embedding(data)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d  = tsne.fit_transform(embeddings)
    

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(data['class'].unique()))

    for idx, class_name in enumerate(data['class'].unique()):
        class_idx = get_class_idx(class_name)
        indices = labels == class_idx
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    label=class_name, alpha=0.6, s=30, color=palette[class_idx])

    for i in range(len(points)):
        x, y = embeddings_2d[len_data+i, 0], embeddings_2d[len_data+i, 1]
        plt.scatter(x, y, color='black', marker='X', s=100, edgecolors='black')
        plt.text(x + 1.5, y, tasks[i], fontsize=12, weight='bold', color='black')

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tSNE_Plot()