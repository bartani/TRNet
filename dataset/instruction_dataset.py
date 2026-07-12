import pandas as pd
from torch.utils.data import Dataset, DataLoader
import dataset_configeration as config
import random
import os

class train_dataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.points = config.get_Points()
        _, idx_class = config.get_class_points(self.points, df['class'])
        df['idx_class'] = idx_class
        self.class_to_indices = df.groupby('class').apply(lambda x: x.index.tolist()).to_dict()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pos_row = self.df.iloc[index]
        pos_text = pos_row['text']
        pos_label = pos_row['class']
        pos_idx = pos_row['idx_class']

        anc = self.points[pos_idx]

        negative_label = random.choice([c for c in self.class_to_indices if c != pos_label])
        neg_idx = random.choice(self.class_to_indices[negative_label])
        negative_text = self.df.iloc[neg_idx]['text']
        negative_lbl = self.df.iloc[neg_idx]['class']



        return {
            'anc': anc,
            'pos_text': pos_text,
            'label': pos_label,
            'neg_text': negative_text,
            'pos_idx' : pos_idx,
            'negative_lbl' : negative_lbl,
        }


def train_loader():
    file_path = os.path.join(".", "data", "7-train_instruction_translated.csv")
    df = pd.read_csv(file_path)
    myds = train_dataset(df)
    loader = DataLoader(
        myds,
        batch_size=config.instruction_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader

if __name__ == "__main__":
    loader = train_loader()
    
    d = next(iter(loader))
    labels = d['label']
    idx_point = d['pos_idx']
    neg = d['negative_lbl']
    anc = d['anc']
    print(labels)
    print(idx_point.shape)
    print(anc.shape)
    print("-------------------------------")
    print(anc)
    print("-------------------------------")
    print(config.get_Points())
    print("-------------------------------")
    print(d['pos_text'])

    

    