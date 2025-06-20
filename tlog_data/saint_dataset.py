import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SAINTDataset(Dataset):
    def __init__(self, df, num_cols=None, cat_cols=None, label_col='loginfo'):
        super().__init__()
        
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        
        df['loginfo'] = df['loginfo'].map({'normal': 0, 'fail': 1})
        self.org_df = df
        self.labels = self.org_df["loginfo"].values.astype('float32')
        self.df = self.org_df.drop(columns=['timestamp', 'TimeUS', label_col], errors='ignore')
        

        # 범주형/연속형 데이터가 있는지 확인하고, 없을 때의 처리
        if len(self.cat_cols) > 0:
            self.cat_data = self.df[cat_cols].astype('category') \
                .apply(lambda x: x.cat.codes).values.astype('int64')
        else:
            self.cat_data = None

        if len(self.num_cols) > 0:
            self.num_data = self.df[num_cols].values.astype('float32')
        else:
            self.num_data = None

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 범주형: 없으면 빈 텐서로 반환
        if self.cat_data is not None:
            x_cat = torch.tensor(self.cat_data[idx], dtype=torch.long)
        else:
            x_cat = torch.empty(0, dtype=torch.long)  # <-- 빈 텐서

        # 수치형: 없으면 빈 텐서로 반환
        if self.num_data is not None:
            x_num = torch.tensor(self.num_data[idx], dtype=torch.float)
        else:
            x_num = torch.empty(0, dtype=torch.float)  # <-- 빈 텐서

        y = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)

        return x_cat, x_num, y
        
