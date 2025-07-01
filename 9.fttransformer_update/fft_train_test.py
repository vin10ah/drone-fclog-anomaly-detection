import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Dataloader_fft import TabularDataset, SequenceTabularDataset
from FTTransform import *
from FTTransform2 import *
from FTTransform3 import *
from preprocess_func import preprocess_func
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt 
from Earlystopping import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import joblib
import onnx
from plattscaler import PlattScaler
from threshold import *



def find_best_threshold_youden(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def find_generalized_threshold(model_class, dataset, device, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = np.array([label for _, label in dataset])
    thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n[KFold] Fold {fold+1}/{k}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

        model = model_class(num_numerical_features=len(
            
        )).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(5), desc=f"Training Fold {fold+1}"):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).float()
                optimizer.zero_grad()
                output = model(x).squeeze(1)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                output = model(x).squeeze(1)
                all_logits.extend(output.cpu().numpy())
                all_labels.extend(y.numpy())

        platt = PlattScaler()
        platt.fit(np.array(all_logits), np.array(all_labels))
        probs = platt.predict_proba(np.array(all_logits))[:,1]
        best_thr = find_best_threshold_youden(np.array(all_labels), probs)
        print(f"  [Fold {fold+1}] Best Threshold (Youden): {best_thr:.4f}")
        thresholds.append(best_thr)

    median_thr = np.median(thresholds)
    print(f"\nGeneralized Threshold (median of folds): {median_thr:.4f}")
    return median_thr


# 디바이스 정의 및 데이터 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = './gaion_train.csv'
numerical_cols = ['roll_ATTITUDE', 'pitch_ATTITUDE', 'xacc_RAW_IMU', 'yacc_RAW_IMU', 'zacc_RAW_IMU', 'zmag_RAW_IMU']
X_num_train, X_num_val, y_train, y_val, scaler = preprocess_func(data_path, numerical_cols)


# 데이터로더 정의
seq_len = 10
train_dataset = SequenceTabularDataset(X_num_train, y_train, seq_len=seq_len)
val_dataset = SequenceTabularDataset(X_num_val, y_val, seq_len=seq_len)
train_loader = DataLoader(train_dataset, batch_size=128)
val_loader = DataLoader(val_dataset, batch_size=128)


# Threshold 일반화
generalized_thr = find_generalized_threshold(
    lambda num_numerical_features: FTTransformerTimeSeries(num_numerical_features=num_numerical_features),
    train_dataset,
    numerical_cols,
    device
)


# 모델 정의
model = FTTransformerTimeSeries(num_numerical_features=len(numerical_cols)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

def train_epoch(loader):
    model.train()
    total_loss = 0
    for x_num, y in loader:
        x_num, y = x_num.to(device), y.to(device).float()
        optimizer.zero_grad()
        output = model(x_num).squeeze(1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ⭐ evaluate()에서 PlattScaler 적용
def evaluate(loader, fixed_thr=None, platt_scaler=None):
    model.eval()
    total_loss = 0
    all_logits, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x_num, y in loader:
            x_num, y = x_num.to(device), y.to(device)
            output = model(x_num).squeeze(1)
            loss = criterion(output, y.float())
            total_loss += loss.item() * x_num.size(0)

            probs = torch.sigmoid(output)
            all_logits.append(output.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Platt scaling 적용
    if platt_scaler:
        calibrated_probs = platt_scaler.predict_proba(all_logits)[:,1]
    else:
        calibrated_probs = all_probs

    if fixed_thr is None:
        best_thr = find_best_threshold_youden(all_labels, calibrated_probs)
    else:
        best_thr = fixed_thr

    preds = (calibrated_probs > best_thr).astype(int)

    acc = (preds == all_labels).mean()
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, calibrated_probs)
    recall = recall_score(all_labels, preds, zero_division=0)
    precision = precision_score(all_labels, preds, zero_division=0)

    print(f"📌 Threshold (Platt 적용 여부: {platt_scaler is not None}): {best_thr:.2f}")

    return total_loss / len(loader.dataset), acc, f1, auc, recall, precision, best_thr, all_logits, all_labels


# 학습 루프
EPOCHS = 200
early_stopping = EarlyStopping(patience=10, verbose=True)
model_name = model._get_name()
print(f"사용하는 장치: {device}")

train_loss_epoch = []
val_loss_epoch = []
f1_epoch = []
best_logits = []
best_labels = []

for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_loss, val_acc, f1, auc, recall, precision, best_thr, logits, labels = evaluate(val_loader, fixed_thr=generalized_thr)

    train_loss_epoch.append(train_loss)
    val_loss_epoch.append(val_loss)
    f1_epoch.append(f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f}")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        best_logits = logits
        best_labels = labels
        print("Early stopping으로 학습 종료")
        break

platt_scaler = PlattScaler()
platt_scaler.fit(best_logits, best_labels)
val_loss, val_acc, f1, auc, recall, precision, best_thr, _, _ = evaluate(val_loader, fixed_thr=generalized_thr, platt_scaler=platt_scaler)



# 모델 저장
model.load_state_dict(early_stopping.best_state_dict)
MODEL_DIR = './models'
MODEL_PATH = f"{MODEL_DIR}/{model_name}({data_path[2:-4]}).pth"
ONNX_PATH = f"{MODEL_DIR}/{model_name}({data_path[2:-4]}).onnx"
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save({'model_state_dict': model.state_dict(), 'best_thr': best_thr}, MODEL_PATH)
print(f"모델 저장 완료: {MODEL_PATH}")

dummy_input = torch.randn(1, seq_len, len(numerical_cols)).to(device)
torch.onnx.export(model, dummy_input, ONNX_PATH, export_params=True, opset_version=17,
                  do_constant_folding=True, input_names=['input'], output_names=['output'])
print(f"ONNX 모델 저장 완료: {ONNX_PATH}")


# 데이터 스케일러 저장
SCALER_DIR = f'./scaler/{model_name}'
SCALER_SAVE_NAME = f'/scaler_{model_name}({data_path[2:-4]}).pkl'

os.makedirs(SCALER_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_DIR + SCALER_SAVE_NAME)

print(f"스케일러 저장 완료: {SCALER_DIR + SCALER_SAVE_NAME}")


# PlattScaler 저장 (확률보정)
SCALER_DIR = f'./scaler/{model_name}'
PLATT_PATH = f'{SCALER_DIR}/platt_scaler_{model_name}({data_path[2:-4]}).pkl'

joblib.dump(platt_scaler, PLATT_PATH)
print(f"Platt Scaler가 '{PLATT_PATH}' 경로에 저장되었습니다.")


# 평가 및 시각화
model.eval()
all_preds, all_labels = [], []
all_logits = []

with torch.no_grad():
    for x_num, y in val_loader:
        x_num = x_num.to(device)
        output = model(x_num).squeeze(1)
        logits = output.cpu().numpy()
        probs = platt_scaler.predict_proba(logits)[:,1]
        preds = (probs > best_thr).astype(int)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())


# 혼동행렬 및 그래프
PLOT_SAVE_DIR = './result_plots'
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix')
plt.savefig(os.path.join(PLOT_SAVE_DIR, f'confusion_matrix_{model_name}({data_path[2:-4]}).png'))
plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_epoch, label='Train Loss')
plt.plot(val_loss_epoch, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(f1_epoch, label='F1 Score', color='green')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_SAVE_DIR, f'training_curves_{model_name}({data_path[2:-4]}).png'))
plt.close()

print(f"\n📊 시각화 및 혼동행렬 저장 완료: '{PLOT_SAVE_DIR}'")
