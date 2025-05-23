import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def compute_threshold(model, trn_tensor, quantile):
    model.eval()
    with torch.no_grad():
        trn_tensor = trn_tensor.to(device)
        recon = model(trn_tensor)
        loss_each = ((trn_tensor - recon) ** 2).mean(dim=1)
        threshold = torch.quantile(loss_each, quantile)
    return threshold.item()

def train(trn_loader, **params):
    save_dir = './results/models'
    os.makedirs(save_dir, exist_ok=True)

    print(f'device: {device}')
    lr = params.get('lr')
    epochs = params.get('epochs')
    batch_size = params.get('batch_size')

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

    filename = f"ae_{batch_size}_lr{lr:.0e}_ep{epochs}"
    torch.save(model.state_dict(), f'{save_dir}/{filename}.pth')

    return model, loss_lst

def predict(model, tst_tensor, threshold):
    model.eval()
    with torch.no_grad():
        tst_tensor = tst_tensor.to(device)
        recon = model(tst_tensor)
        recon_error = ((tst_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        y_pred = (recon_error > threshold).astype(int)
    return recon_error, y_pred


##### 성능 csv로 저장 #####

import csv

def save_report_to_csv(report_normal, report_abnormal, **params):
    batch_size = params.get('batch_size')
    lr = params.get('lr')
    quantile = params.get('quantile')
    epochs = params.get('epochs')

    save_path = './results/performance_summary.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def extract_line_metrics(report, label_name):
        lines = report.strip().split('\n')
        for line in lines:
            if line.strip().startswith(label_name):
                parts = line.split()
                return parts[1], parts[2], parts[3]  # precision, recall, f1-score
        return "0", "0", "0"

    nor_p, nor_r, nor_f1 = extract_line_metrics(report_normal, "Normal")
    ab_p, ab_r, ab_f1 = extract_line_metrics(report_abnormal, "Anomaly")

    header = [
        'batch_size', 'lr', 'quantile', 'epochs',
        'nor_precision', 'nor_recall', 'nor_f1',
        'ab_precision', 'ab_recall', 'ab_f1'
    ]
    row = [
        batch_size, lr, quantile, epochs,
        nor_p, nor_r, nor_f1,
        ab_p, ab_r, ab_f1
    ]

    write_header = not os.path.exists(save_path)
    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Report saved to CSV: {save_path}")


##### Confusion Matices 저장 #####

def compare_confusion_matrices(y_true_normal, y_pred_normal, y_true_abnormal, y_pred_abnormal, **params):
    batch_size = params.get('batch_size')
    lr = params.get('lr')
    quantile = params.get('quantile')
    epochs = params.get('epochs')

    save_dir = './results/confusion_matrices'
    os.makedirs(save_dir, exist_ok=True)

    cm_normal = confusion_matrix(y_true_normal, y_pred_normal)
    cm_abnormal = confusion_matrix(y_true_abnormal, y_pred_abnormal)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_normal, display_labels=["Normal", "Anomaly"])
    disp1.plot(ax=axs[0], cmap='YlGnBu', colorbar=False)
    axs[0].set_title("Confusion Matrix (Normal)")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_abnormal, display_labels=["Normal", "Anomaly"])
    disp2.plot(ax=axs[1], cmap='YlGnBu', colorbar=False)
    axs[1].set_title("Confusion Matrix (Abnormal)")

    suptitle = f"Confusion Matrices\n(Batch Size: {batch_size}, LR: {lr}, Quantile: {quantile}, Epochs: {epochs})"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()

    filename = f"cm_bs{batch_size}_lr{lr:.0e}_q{quantile:.2f}_ep{epochs}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

    report_normal = classification_report(y_true_normal, y_pred_normal, target_names=["Normal", "Anomaly"], zero_division=0)
    report_abnormal = classification_report(y_true_abnormal, y_pred_abnormal, target_names=["Normal", "Anomaly"], zero_division=0)

    print("=== 정상 성능 지표 ===")
    print(report_normal)
    print("=== 이상 성능 지표 ===")
    print(report_abnormal)

    # CSV 저장 추가!
    save_report_to_csv(report_normal, report_abnormal, **params)


##### Error 그래프 정상/이상 #####

def plot_mse_comparison(normal_error, abnormal_error, threshold, **params):
    batch_size = params.get('batch_size')
    lr = params.get('lr')
    quantile = params.get('quantile')
    epochs = params.get('epochs')

    save_dir = './results/recon_mse'
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axs[0].plot(normal_error, label='Reconstruction MSE')
    axs[0].axhline(y=threshold, color='blue', linestyle='--', label='Threshold')
    axs[0].set_title('Normal Error')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('MSE')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(abnormal_error, label='Reconstruction MSE')
    axs[1].axhline(y=threshold, color='blue', linestyle='--', label='Threshold')
    axs[1].set_title('Abnormal Error')
    axs[1].set_xlabel('time')
    axs[1].legend()
    axs[1].grid(True)

    suptitle = f"Reconstruction Error\n(Batch Size: {batch_size}, LR: {lr}, Quantile: {quantile}, Epochs: {epochs})"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()

    filename = f"recon_errors_bs{batch_size}_lr{lr:.0e}_q{quantile:.2f}_ep{epochs}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


##### loss 그래프 ######

def plot_training_loss(loss_list, batch_size, lr, quantile, epochs, save_dir='./results/training_loss'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Loss')
    suptitle = f"Training Loss\n(Batch Size: {batch_size}, LR: {lr}, Quantile: {quantile}, Epochs: {epochs})"
    plt.suptitle(suptitle, fontsize=12, y=1.03)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f"training_loss_bs{batch_size}_lr{lr:.0e}_q{quantile:.2f}_ep{epochs}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved to: {filepath}")


##### Run All #####

def main():
    df1 = pd.read_csv('../0.data/results/XKF1_merged.csv')
    df2 = pd.read_csv('../0.data/results/XKF2_merged.csv')

    select_col1 = ['Roll', 'Pitch', 'GX', 'GY', 'GZ']
    select_col2 = ['AX', 'AY', 'AZ', 'label']
    concat_df = pd.concat([df1[select_col1], df2[select_col2]], axis=1)

    nor_df = concat_df.loc[concat_df['label'] == 0]
    ab_df = concat_df.loc[concat_df['label'] == 1]

    nor_trn, nor_tst = train_test_split(nor_df, test_size=0.2, random_state=42)
    nor_tst = nor_tst.drop(columns=['label'])
    ab_tst = ab_df.sample(n=len(nor_tst), random_state=42).drop(columns=['label'])
    X_trn = nor_trn.drop(columns=['label'])

    params = {
        'batch_size': 256,
        'lr': 1e-5,
        'quantile': 0.80,
        'epochs': 500
    }

    scaler = MinMaxScaler()
    scaled_X_trn = scaler.fit_transform(X_trn)
    X_trn_tensor = torch.tensor(scaled_X_trn, dtype=torch.float32)
    trn_dataset = TensorDataset(X_trn_tensor)
    trn_loader = DataLoader(trn_dataset, batch_size=params['batch_size'], shuffle=True)

    trained_model, trn_loss_lst = train(trn_loader, **params)
    threshold = compute_threshold(trained_model, X_trn_tensor, quantile=params['quantile'])
    print(f"Threshold ({int(params['quantile'] * 100)}% quantile): {threshold:.6f}")

    nor_tst_tensor = torch.tensor(scaler.transform(nor_tst), dtype=torch.float32)
    ab_tst_tensor = torch.tensor(scaler.transform(ab_tst), dtype=torch.float32)

    nor_recon_error, nor_y_pred = predict(trained_model, nor_tst_tensor, threshold)
    nor_y_true = np.zeros_like(nor_y_pred)

    ab_recon_error, ab_y_pred = predict(trained_model, ab_tst_tensor, threshold)
    ab_y_true = np.ones_like(ab_y_pred)

    compare_confusion_matrices(nor_y_true, nor_y_pred, ab_y_true, ab_y_pred, **params)
    plot_mse_comparison(nor_recon_error, ab_recon_error, threshold, **params)
    plot_training_loss(trn_loss_lst, params['batch_size'], params['lr'], params['quantile'], params['epochs'])

if __name__ == "__main__":
    main()

