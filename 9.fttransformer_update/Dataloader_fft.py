import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, X_num, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx],  self.y[idx]
    
class test_dataset(Dataset):
    def __init__(self,X_num):
        self.test_X_num = torch.tensor(X_num, dtype=torch.float32)
        
    def __len__(self):
        return len(self.test_X_num)    
    def __getitem__(self, idx):
        X = self.test_X_num[idx]
        X = X.unsqueeze(0).unsqueeze(0) # -> (1,1,F)
        return X
class SequenceTabularDataset(Dataset):
    def __init__(self, X_num, y, seq_len=10):
        self.seq_len = seq_len
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.valid_len = len(self.X_num) - seq_len + 1

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        x_seq = self.X_num[idx : idx + self.seq_len]        # (seq_len, num_features)
        y_label = self.y[idx + self.seq_len - 1]            # 시퀀스의 마지막 시점 라벨
        return x_seq, y_label
    
class FTTransformerTestDataset(Dataset):
    def __init__(self, X_num, seq_len=10):
        self.seq_len = seq_len
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.valid_len = len(self.X_num) - seq_len + 1  # 시퀀스 만들 수 있는 구간

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        x_seq = self.X_num[idx : idx + self.seq_len]   # (seq_len, feature)
        x_seq = x_seq.unsqueeze(0)      # ( 1, seq_len, feature)
        return x_seq
