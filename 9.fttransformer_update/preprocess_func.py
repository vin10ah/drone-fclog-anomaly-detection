import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Dataloader_fft import *
from torch.utils.data import DataLoader

def preprocess_func(data_url, numerical_cols):
    # 데이터 로딩
    df = pd.read_csv(data_url)

    # 피처 구분
    cols = numerical_cols
    target_col = 'label'

    # 수치형 스케일링
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    # 데이터 분할
    X_num = df[cols].values
    y = df[target_col].values

    X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_num, y, test_size=0.2, random_state=42
    )
    return X_num_train, X_num_val, y_train, y_val , scaler

def preprocess_for_npu(model_type=None,scaler=None,tst_csv_url=None,columns=None):
    test_csv = pd.read_csv(tst_csv_url)
    test_csv = test_csv[columns]
    scaled_test_data = scaler.transform(test_csv)
    if model_type == 'time':
        test_data = FTTransformerTestDataset(scaled_test_data)
        test_loader = DataLoader(test_data, batch_size=128)
    else:
        test_data = test_dataset(scaled_test_data)
        test_loader = DataLoader(test_data, batch_size=128)
    return test_loader
    
   