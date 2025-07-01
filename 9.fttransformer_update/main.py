import torch
from preprocess_func import preprocess_func
from Dataloader_fft import TabularDataset, SequenceTabularDataset
from torch.utils.data import Dataset, DataLoader
from threshold import *
import numpy as np
from FTTransform import *
from FTTransform2 import *
from FTTransform3 import *
from Earlystopping import EarlyStopping
import os
from train_eval import *
from threshold import find_generalized_threshold
from train_eval import train_loop, evaluate
from FTTransform2 import FTTransformer2

import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd

# 디바이스 정의 및 데이터 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
''' 
[field_name] 
tlog
AHR2 ATT BAT CTRL CTUN FTN1 FTN2 
IMU MAG MCU POWR RATE VIBE XKF1 
XKF3 XKF4 XKF5 XKV1 XKV2 XKY1
'''
field_name = "AHR2"
selected_feature_df = pd.read_csv("../0.data/selected_features_data.csv")

data_path = f"../0.data/results/{field_name}_merged.csv"

data_batch_size = 128

# field_name = os.path.basename(data_path).split('_')[0]
# numerical_cols = ['roll_ATTITUDE', 'pitch_ATTITUDE', 'xacc_RAW_IMU', 'yacc_RAW_IMU', 'zacc_RAW_IMU', 'zmag_RAW_IMU']
# if selected_feature_df is not None and field_name in selected_feature_df['msg_field'].values:

feature_str = selected_feature_df.loc[selected_feature_df['field'] == field_name, 'feature_list'].values[0]
numerical_cols = [x.strip() for x in feature_str.split(",")]
num_numerical_features=len(numerical_cols)

X_num_train, X_num_val, y_train, y_val, scaler = preprocess_func(data_path, numerical_cols)
dataset_name = os.path.basename(data_path).split('.')[0]

# 데이터로더 정의
seq_len = 10
trn_dataset = TabularDataset(X_num_train, y_train)
val_dataset = TabularDataset(X_num_val, y_val)
trn_loader = DataLoader(trn_dataset, batch_size=data_batch_size)
val_loader = DataLoader(val_dataset, batch_size=data_batch_size)


# 모델 정의
model = FTTransformer2(num_numerical_features).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Threshold 일반화
generalized_thr = find_generalized_threshold(
    lambda num_numerical_features: FTTransformer2(num_numerical_features),
    trn_dataset,
    numerical_cols,
    device,
    k = 2
)

# 학습 루프
epochs = 200
early_stopping = EarlyStopping(patience=10, verbose=True)
model_name = model._get_name()
print(f"사용하는 장치: {device}")
file_name = f'{dataset_name}_{model_name}'

train_dict = {
    'model': model,
    'optimizer': optimizer, 
    'criterion': criterion, 
    'device': "cuda" if torch.cuda.is_available() else "cpu", 
    'trn_loader': trn_loader, 
    'val_loader': val_loader, 
    'epochs' : 1, 
    'generalized_thr': generalized_thr,
    'early_stopping': EarlyStopping(patience=10, verbose=True),
    'file_name': f'{file_name}'
}

model, best_logits, best_labels = train_loop(**train_dict)

platt_scaler = PlattScaler()
platt_scaler.fit(best_logits, best_labels)
val_loss, val_acc, f1, auc, recall, precision, best_thr, _, _ = evaluate(model, criterion, device, val_loader, fixed_thr=generalized_thr, platt_scaler=platt_scaler)


# 데이터 스케일러 저장
SCALER_DIR = f'./results/scaler'
SCALER_SAVE_NAME = f'/{file_name}_scaler.pkl'

os.makedirs(SCALER_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_DIR + SCALER_SAVE_NAME)

print(f"스케일러 저장 완료: {SCALER_DIR + SCALER_SAVE_NAME}")

PLATT_PATH = f'{SCALER_DIR}/{file_name}_platt_scaler.pkl'

joblib.dump(platt_scaler, PLATT_PATH)
print(f"Platt Scaler가 '{PLATT_PATH}' 경로에 저장되었습니다.")
