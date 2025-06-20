import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def sigmoid_preds(logits):
    probs = torch.sigmoid(logits)
    return (probs > 0.5).long()


def evaluate_binary_classification(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    
    for x_cat, x_num, y in tqdm(dataloader, desc="Training", leave=False):
        x_cat = x_cat.to(device) if x_cat is not None else None
        x_num = x_num.to(device) if x_num is not None else None
        y = y.float().to(device)  # BCE는 float

        optimizer.zero_grad()
        outputs = model(x_cat, x_num)  # (B,)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

        preds = sigmoid_preds(outputs)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = evaluate_binary_classification(y_true, y_pred)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics



def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_cat, x_num, y in tqdm(dataloader, desc="Validation", leave=False):
            x_cat = x_cat.to(device) if x_cat is not None else None
            x_num = x_num.to(device) if x_num is not None else None
            y = y.float().to(device)

            outputs = model(x_cat, x_num)
            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)

            preds = sigmoid_preds(outputs)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = evaluate_binary_classification(y_true, y_pred)
    avg_loss = total_loss / len(dataloader.dataset)

    
    return avg_loss, metrics, y_pred, y_true

def train_valid(test, model, optimizer, criterion, device, trn_loader, val_loader, epochs):
    best_val_loss = float('inf')
    os.makedirs("./results", exist_ok=True)
    best_metrics = {}
    best_epoch = 0

    early_stopper = EarlyStopping(patience=10, verbose=True)


    for epoch in range(epochs):
        trn_loss, trn_metrics = train_one_epoch(model, trn_loader, optimizer, criterion, device)
        val_loss, val_metrics, preds, labels = validate(model, val_loader, criterion, device)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            best_preds = preds
            best_labels = labels
            best_epoch = epoch + 1
            
            torch.save(model.state_dict(), f"./results/tlog_{test}_SAINT_best_model.pt")
            print(f"✅ Best model saved at epoch {epoch+1} (val_loss={val_loss:.4f})")

        print(f"[Epoch {epoch+1}]")
        print(f"Train Loss: {trn_loss:.4f} | "
            f"F1: {trn_metrics['f1']:.4f}, "
            f"Precision: {trn_metrics['precision']:.4f}, "
            f"Recall: {trn_metrics['recall']:.4f}, "
            f"Acc: {trn_metrics['accuracy']:.4f}")

        print(f"Val Loss: {val_loss:.4f} | "
            f"F1: {val_metrics['f1']:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping 체크
        early_stopper.step(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
        
    # Save metrics to TXT
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/tlog_{test}_metrics.txt", "w") as f:
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"val_loss: {best_val_loss:.4f}\n")
        for k, v in best_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # Save confusion matrix
    os.makedirs("./results", exist_ok=True)
    cm = confusion_matrix(best_labels, best_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix")
    plt.savefig(f"./results/tlog_{test}_confmat.png")
    plt.close()