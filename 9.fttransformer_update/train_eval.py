import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from threshold import *
from Earlystopping import EarlyStopping
import os
import joblib

def train_epoch(model, optimizer, criterion, device, loader):
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
def evaluate(model, criterion, device, loader, fixed_thr=None, platt_scaler=None):
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


def train_loop(model,
               optimizer, 
               criterion, 
               device, 
               trn_loader, 
               val_loader, 
               epochs, 
               generalized_thr,
               early_stopping,
               file_name
               ):

    MODEL_DIR = './results/models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_loss_epoch = []
    val_loss_epoch = []
    f1_epoch = []
    best_logits = []
    best_labels = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, criterion, device, trn_loader)
        val_loss, val_acc, f1, auc, recall, precision, best_thr, logits, labels = evaluate(
            model, criterion, device, val_loader, fixed_thr=generalized_thr)

        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        f1_epoch.append(f1)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | "
              f"Recall: {recall:.4f} | Precision: {precision:.4f}")

        # 🔑 Best Loss 기준으로 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_thr': best_thr
            }, os.path.join(MODEL_DIR, f"{file_name}_best_{epoch+1}e.pt"))
            print(f">> [Best Saved] val_loss improved: {best_val_loss:.4f}")

            # ✅ best ONNX도 저장
            onnx_path = os.path.join(MODEL_DIR, f"{file_name}_best_{epoch+1}e.onnx")
            dummy_input = torch.randn(1, model.num_numerical_features).to(device)
            torch.onnx.export(model, dummy_input, onnx_path,
                            export_params=True,
                            opset_version=17,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output'])
            print(f"📦 ONNX 저장됨: {onnx_path}")

            best_logits = logits
            best_labels = labels


        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("***** Early stopping으로 학습 종료 *****")
            break

    # 🔑 마지막 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_thr': best_thr
    }, os.path.join(MODEL_DIR, f"{file_name}_last_{epoch+1}e.pt"))
    print(f"💾 [Last Saved] 최종 epoch 모델 저장 완료")


    return model, best_logits, best_labels