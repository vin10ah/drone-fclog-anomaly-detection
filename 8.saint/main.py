import numpy as np
import pandas as pd
import os
import glob
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from saint_simple import SAINTModel
from saint_dataset import SAINTDataset
from train import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
epochs = 100



selected_df = pd.read_csv("../0.data/selected_features.csv")
msg_lst = list(selected_df["msg_field"])
# msg_lst = sorted(glob.glob("../0.data/results/*.csv"))

for msg in msg_lst:

    os.makedirs("./results/model", exist_ok=True)
    
    selected_cols = selected_df.loc[selected_df['msg_field']==msg, 'feature_list'].values[0].split(', ')
    selected_cols.append('label')
    
    csv_path = f"../0.data/results/{msg}_merged.csv"
    df = pd.read_csv(csv_path)[selected_cols]
    
    trn_df, val_df = train_test_split(df, test_size=0.3, shuffle=True, stratify=df['label'], random_state=42)

    # label 제외 스케일링
    num_cols = [col for col in selected_cols if col != 'label']
    scaler = StandardScaler()
    trn_df[num_cols] = scaler.fit_transform(trn_df[num_cols])
    val_df[num_cols] = scaler.transform(val_df[num_cols])

    # 추론시 동일한 스케일러 사용 가능
    joblib.dump(scaler, f"./results/model/{msg}_scaler.pkl")

    trn_dataset = SAINTDataset(trn_df, num_cols=num_cols, cat_cols=None, label_col='label')
    val_dataset = SAINTDataset(val_df, num_cols=num_cols, cat_cols=None, label_col='label')
    
    trn_loader = DataLoader(trn_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)


    model = SAINTModel(num_categories=[], num_numericals=len(num_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    train_valid(msg, model, optimizer, criterion, device, trn_loader, val_loader, epochs)
    




