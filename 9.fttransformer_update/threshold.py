from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from plattscaler import PlattScaler
from tqdm import tqdm


def find_best_threshold_youden(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def find_generalized_threshold(model_class, dataset, numerical_cols, device, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = np.array([label for _, label in dataset])
    thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n[KFold] Fold {fold+1}/{k}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

        model = model_class(num_numerical_features=len(numerical_cols)).to(device)
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
