import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SAINTDataset(Dataset):
    def __init__(self, csv_path, num_cols=None, cat_cols=None, label_col='label'):
        super().__init__()
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.org_df = pd.read_csv(csv_path)
        self.labels = self.org_df["label"].values.astype('float32')
        self.df = self.org_df.drop(columns=['timestamp', 'TimeUS', label_col], errors='ignore')
        

        # 범주형 데이터가 있는지 확인하고, 없을 때의 처리
        if len(self.cat_cols) > 0:
            self.cat_data = self.df[cat_cols].astype('category') \
                .apply(lambda x: x.cat.codes).values.astype('int64')

        if len(self.num_cols) > 0:
            self.num_data = self.df[num_cols].values.astype('float32')

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x_cat = torch.tensor(self.cat_data[idx], dtype=torch.long) if self.cat_cols is not None else None
        x_num = torch.tensor(self.num_data[idx], dtype=torch.float) if self.num_data is not None else None
        y = torch.tensor(self.labels[idx], dtype=torch.long)
    
        return x_cat, x_num, y
    
