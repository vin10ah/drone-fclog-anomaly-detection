import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# params = {
#     'batch_size': 1024,
#     'epochs': 500,
#     'lr': 1e-5
# }


##### 분포 그래프 그리기 #####

def plot_feature_trend(df, features, label_col='label', title='Feature Trends (Normal vs Abnormal)'):
    num_cols = 4
    num_rows = (len(features) + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    # 레이블 분리
    normal = df[df[label_col] == 0].reset_index(drop=True)
    abnormal = df[df[label_col] == 1].reset_index(drop=True)

    for i, feature in enumerate(features):
        ax = axs[i]

        # x축: 각각 0부터 시작
        x_normal = range(len(normal))
        x_abnormal = range(len(abnormal))

        ax.plot(x_normal, normal[feature], color='blue', alpha=0.6, linewidth=0.7, label='Normal')
        ax.plot(x_abnormal, abnormal[feature], color='red', alpha=0.6, linewidth=0.7, label='Abnormal')

        ax.set_title(f'{feature}', fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'{feature} Value')
        ax.grid(True)
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # 빈 subplot 제거
    for j in range(len(features), len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(title, fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

##### 오토인코더 모델 #####

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out



##### 임계값 찾기 #####

def compute_threshold(model, trn_tensor, quantile=0.95):
    model.eval()
    with torch.no_grad():
        trn_tensor = trn_tensor.to(device)
        recon = model(trn_tensor)
        loss_each = ((trn_tensor - recon) ** 2).mean(dim=1)
        threshold = torch.quantile(loss_each, quantile)

    return threshold.item()



##### 모델 학습 #####

def train(trn_loader, **params):
    print(f'device: {device}')
    lr = params.get('lr', 1e-5)
    epochs = params.get('epochs', 500)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_lst = []

    epoch_bar = tqdm(range(epochs))

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0

        for batch in trn_loader:
            x_batch = batch[0].to(device)

            output = model(x_batch)
            loss = criterion(output, x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(trn_loader)
        loss_lst.append(avg_loss)

        epoch_bar.set_postfix(loss=avg_loss)

    return model, loss_lst



##### 테스트셋 예측 #####

def predict(model, tst_tensor, threshold):
    model.eval()
    with torch.no_grad():
        tst_tensor = tst_tensor.to(device)
        recon = model(tst_tensor)
        recon_error = ((tst_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        y_pred = (recon_error > threshold).astype(int)

    return recon_error, y_pred



##### 예측 평가 #####

def compare_confusion_matrices(y_true_normal, y_pred_normal, y_true_abnormal, y_pred_abnormal, **params):
    batch_size = params.get('batch_size')
    epochs = params.get('epochs')
    lr = params.get('lr')

    save_dir = './results/confusion_matrices'
    os.makedirs(save_dir, exist_ok=True)

    # 혼동 행렬 계산
    cm_normal = confusion_matrix(y_true_normal, y_pred_normal)
    cm_abnormal = confusion_matrix(y_true_abnormal, y_pred_abnormal)

    # 시각화
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # 정상
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_normal, display_labels=["Normal", "Anomaly"])
    disp1.plot(ax=axs[0], cmap='YlGnBu', colorbar=False)
    axs[0].set_title("Confusion Matrix (Normal)")

    # 이상
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_abnormal, display_labels=["Normal", "Anomaly"])
    disp2.plot(ax=axs[1], cmap='YlGnBu', colorbar=False)
    axs[1].set_title("Confusion Matrix (Abnormal)")

    suptitle = f"Confusion Matrices\n(Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {lr})"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    filename = f"cm_bs{batch_size}_ep{epochs}_lr{lr:.0e}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.show()

    # 성능 지표 출력
    print("=== 정상 성능 지표 ===")
    print(classification_report(y_true_normal, y_pred_normal, target_names=["Normal", "Anomaly"], zero_division=0))

    print("=== 이상 성능 지표 ===")
    print(classification_report(y_true_abnormal, y_pred_abnormal, target_names=["Normal", "Anomaly"], zero_division=0))



##### 오차 재현 #####

def plot_mse_comparison(normal_error, abnormal_error, threshold, **params):
    batch_size = params.get('batch_size')
    epochs = params.get('epochs')
    lr = params.get('lr')
    
    save_dir = './results/recon_mse'
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # 정상 데이터
    axs[0].plot(normal_error, label='Reconstruction MSE')
    axs[0].axhline(y=threshold, color='blue', linestyle='--', label='Threshold')
    axs[0].set_title('Normal Error')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('MSE')
    axs[0].legend()
    axs[0].grid(True)

    # 이상 데이터
    axs[1].plot(abnormal_error, label='Reconstruction MSE')
    axs[1].axhline(y=threshold, color='blue', linestyle='--', label='Threshold')
    axs[1].set_title('Abnormal Error')
    axs[1].set_xlabel('time')
    axs[1].legend()
    axs[1].grid(True)

    suptitle = f"Reconstruction Error\n(Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {lr})"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.05)

    plt.tight_layout()

    filename = f"recon_errors_{batch_size}_ep{epochs}_lr{lr:.0e}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.show()


##### Loss 그래프 그리기 #####

def plot_training_loss(loss_list, batch_size, epochs, lr, save_dir='./results/training_loss'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Loss')
    
    suptitle = f"Training Loss\n(Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {lr})"
    plt.suptitle(suptitle, fontsize=12, y=1.03)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"training_loss_bs{batch_size}_ep{epochs}_lr{lr:.0e}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

    print(f"Training loss plot saved to: {filepath}")


def main():
    # 데이터 불러오기
    df1 = pd.read_csv('../0.data/results/XKF1_merged.csv')
    df2 = pd.read_csv('../0.data/results/XKF2_merged.csv')

    select_col1 = ['Roll', 'Pitch', 'GX', 'GY', 'GZ']
    select_col2 = ['AX', 'AY', 'AZ', 'label']

    concat_df = pd.concat([df1[select_col1], df2[select_col2]], axis=1)

    # 라벨로 분리
    nor_df = concat_df.loc[concat_df['label'] == 0]
    ab_df = concat_df.loc[concat_df['label'] == 1]

    # 분할
    nor_trn, nor_tst = train_test_split(nor_df, test_size=0.2, random_state=42)
    nor_tst = nor_tst.drop(columns=['label'])
    ab_tst = ab_df.sample(n=len(nor_tst), random_state=42).drop(columns=['label'])
    X_trn = nor_trn.drop(columns=['label'])

    # 파라미터
    params = {
        'batch_size': 1024,
        'epochs': 500,
        'lr': 1e-3
    }

    # 스케일링
    scaler = MinMaxScaler()
    scaled_X_trn = scaler.fit_transform(X_trn)

    X_trn_tensor = torch.tensor(scaled_X_trn, dtype=torch.float32)
    trn_dataset = TensorDataset(X_trn_tensor)
    trn_loader = DataLoader(trn_dataset, batch_size=params['batch_size'], shuffle=True)

    # 학습
    trained_model, trn_loss_lst = train(trn_loader, **params)

    # 임계값 설정
    q = 0.95
    threshold = compute_threshold(trained_model, X_trn_tensor, q)
    print(f"Threshold ({round(q*100)}% quantile): {threshold:.6f}")

    # 테스트 예측
    nor_tst_tensor = torch.tensor(scaler.transform(nor_tst), dtype=torch.float32)
    ab_tst_tensor = torch.tensor(scaler.transform(ab_tst), dtype=torch.float32)

    nor_recon_error, nor_y_pred = predict(trained_model, nor_tst_tensor, threshold)
    nor_y_true = np.zeros_like(nor_y_pred)

    ab_recon_error, ab_y_pred = predict(trained_model, ab_tst_tensor, threshold)
    ab_y_true = np.ones_like(ab_y_pred)

    # 성능 평가
    compare_confusion_matrices(nor_y_pred, nor_y_true, ab_y_pred, ab_y_true, **params)

    # MSE 시각화
    plot_mse_comparison(
        normal_error=nor_recon_error, 
        abnormal_error=ab_recon_error, 
        threshold=threshold,
        **params
    )

    # train loss 그래프 시각화
    plot_training_loss(
    loss_list=trn_loss_lst,
    batch_size=params['batch_size'],
    epochs=params['epochs'],
    lr=params['lr']
    )


if __name__ == "__main__":
    main()
