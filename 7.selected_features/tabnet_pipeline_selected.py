import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_tabnet.metrics import Metric

import os
import glob
import math


################## save dir 수정 ##################

# 저장 경로 설정
save_dir = "./results/tabnet"
os.makedirs(save_dir, exist_ok=True)

class my_metric(Metric):
    """
    recall
    """

    def __init__(self):
        self._name = "recall" 
        self._maximize = True

    def __call__(self, y_true, y_score):
        # 이진 분류용 처리
        if y_score.ndim > 1 and y_score.shape[1] > 1:
            # softmax 결과 중 class 1 확률만 사용
            y_pred = np.argmax(y_score, axis=1)
        else:
            # sigmoid 출력 기반
            y_pred = (y_score > 0.5).astype(int)

        return recall_score(y_true, y_pred)


# TabNet

class TabNetPipeline:
    def __init__(self, data_path, scaler=None, save_dir=None, params=None): 
        self.data_path = data_path
        self.msg_name = os.path.basename(self.data_path).split("_")[0]
        self.scaler = scaler

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.params = params or self.model_params()
        self.model = TabNetClassifier(**self.model_params())

        self.best_model_dir = os.path.join(self.save_dir, "best_model")
        os.makedirs(self.best_model_dir, exist_ok=True)

        self.feature_df = pd.read_csv("../0.data/selected_features_20250529.csv")

    def model_params(self):
        return {
            "n_d": 64,                         # 디코더에서 사용하는 feature vector의 차원(decision layer output)
            "n_a": 64,                         # attention layer에서 사용하는 feature vector의 차원 (attention embedding)
            "n_steps": 5,                      # 전체 decision step의 수
            "gamma": 1.5,                      # feature reusage 제어 하이퍼파라미터 (큰 값일수록 feature 재사용 억제)
            "n_independent": 2,                # 각 step의 attention block 내에서 독립적으로 학습하는 layer 수
            "n_shared": 2,                     # 여러 step 간에 공유되는 layer 수
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=2e-2),
            "mask_type": "entmax",             # feature selection 시 사용하는 마스킹 함수 'entmax' 또는 'sparsemax'(default)
            "verbose": 5,                      
            "seed": 42,
            # "loss_fn" :  loss_fn,
            # "output_dim": 1,
            "device_name": 'cuda' if torch.cuda.is_available() else 'cpu',
        }
    
    def train_params(self, custom_params=None):

        default_params = {
            "X_train": self.X_scaled["train"], 
            "y_train": self.y["train"],
            "eval_set": [(self.X_scaled["val"], self.y["val"])],
            "eval_name": ["valid"],
            "eval_metric": [my_metric, "logloss"],
            "max_epochs": 150,
            "patience": 10,
            "batch_size": 2048,
            "virtual_batch_size": 1024,
            "num_workers": 6,
            "drop_last": False,    
            # "save_checkpoint": True,
            # "log_path": self.best_model_dir,
            # "weights_name": f"{self.msg_name}_tabnet_best_model"            
        }

        if custom_params:
            default_params.update(custom_params)
        
        return default_params

    def load_prepare_data(self):
        feature_lst = self.feature_df.loc[self.feature_df["msg_field"] == self.msg_name, "feature_list"].values[0].split(", ")
        feature_lst.append("label")

        df = pd.read_csv(self.data_path)
        df = df.drop(columns=["timestamp", "TimeUS"], axis=1)
        df = df.dropna(axis=1, how='all')  # 값이 전부 NaN인 컬럼 제거(CTUN)

        n_df = df[feature_lst]
        print(n_df.columns)

        X = n_df.drop(columns=["label"])
        y = n_df["label"]

        X_trn_val, X_tst, y_trn_val, y_tst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_trn, X_val, y_trn, y_val = train_test_split(X_trn_val, y_trn_val, test_size=0.2, random_state=42, stratify=y_trn_val)

        self.X = {"train": X_trn, "val": X_val, "test": X_tst}
        self.y = {"train": y_trn.astype(np.float32), 
                  "val": y_val.astype(np.float32), 
                  "test": y_tst.astype(np.float32)
                  }

        self.X_scaled = {
            "train": self.scaler.fit_transform(X_trn),
            "val": self.scaler.transform(X_val),
            "test": self.scaler.transform(X_tst),
        }

    def train(self, custom_params=None):
        self.model.fit(**self.train_params(custom_params))
        self.history = self.model.history

        model_path = os.path.join(self.best_model_dir, f"{self.msg_name}(selected)_tabnet_best_model")
        self.model.save_model(model_path)


    def val_evaluate(self, save_csv=False):
        preds = self.model.predict(self.X_scaled["val"])
        probs = self.model.predict_proba(self.X_scaled["val"])[:, 1]

        best_epoch = np.argmin(self.history["valid_logloss"]) + 1
        
        metrics = {
            "msg_name": self.msg_name,
            "accuracy": round(accuracy_score(self.y["val"], preds), 4),
            "precision": round(precision_score(self.y["val"], preds), 4),
            "recall": round(recall_score(self.y["val"], preds), 4),
            "f1": round(f1_score(self.y["val"], preds), 4),
            "roc_auc": round(roc_auc_score(self.y["val"], probs), 4),
            "best_epoch": best_epoch
        }


        print(f"[{self.msg_name}] Validation Metrics")
        for k, v in metrics.items():
            if k != "msg_name":
                print(f"{k.capitalize():10}: {v:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y["val"], preds))
        print("\nClassification Report:")
        print(classification_report(self.y["val"], preds))

        # CSV 저장
        if save_csv:
            # results_dir = os.path.join(self.save_dir, "results")
            # os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(self.save_dir, "val_metrics.csv")

            df = pd.DataFrame([metrics])
            if os.path.exists(result_file):
                df_existing = pd.read_csv(result_file)
                df = pd.concat([df_existing, df], ignore_index=True)
            df.to_csv(result_file, index=False)
            print(f"[{self.msg_name}] Metrics saved to {result_file}")

    def run(self):
        self.load_prepare_data()
        self.train()
        self.val_evaluate()

    




# 모델 학습 결과 시각화 (Validation data) 

class ResultsVisualizer:
    def __init__(self, pipeline: TabNetPipeline):
        self.pipeline = pipeline
        self.msg_name = pipeline.msg_name
        self.save_dir = pipeline.save_dir

        self.best_model_path = os.path.join(pipeline.best_model_dir, f"{self.msg_name}(selected)_tabnet_best_model.zip")
        pipeline.model.load_model(self.best_model_path)
        self.model = pipeline.model

        self.history = self.model.history
        self.X_scaled = pipeline.X_scaled
        self.y = pipeline.y
        self.feature_names = pipeline.X["val"].columns.tolist()

        self.viz_dir = os.path.join(self.save_dir, "visualizations")
        self.mask_dir = os.path.join(self.viz_dir, "masks")
        self.imp_dir = os.path.join(self.viz_dir, "importance")
        self.loss_dir = os.path.join(self.viz_dir, "loss")

        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.imp_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)

        self.explain_matrix, self.masks = self.model.explain(self.X_scaled["val"])


    def plot_confusion_matrix(self):
        preds = self.model.predict(self.X_scaled["val"])
        cm = confusion_matrix(self.y["val"], preds)
        labels = [0, 1]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        title_font = {
            "fontsize": 16,
            "fontweight": "bold"
        }
        plt.title(f"{self.msg_name}(selected) Confusion Matrix", fontdict=title_font, pad=10)
        
        cm_dir = os.path.join(self.viz_dir, "confusion_matrix")
        os.makedirs(cm_dir, exist_ok=True)
        plt.savefig(os.path.join(cm_dir, f"{self.msg_name}(selected)_confusion_matrix.png"))
        plt.close()


    def plot_masks(self):
        import math

        n_masks = len(self.masks)

        # 마스크 수에 따라 열 수 자동 조정 (최대 4)
        if n_masks <= 3:
            n_cols = n_masks
        elif n_masks <= 6:
            n_cols = 3
        else:
            n_cols = 4  # 많은 경우 가로 넓게

        n_rows = math.ceil(n_masks / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)

        for i in range(n_masks):
            row, col = divmod(i, n_cols)
            ax = axs[row][col]
            mask = self.masks[i]

            if mask is None or np.asarray(mask).ndim != 2:
                ax.axis("off")
                continue

            im = ax.imshow(np.asarray(mask), aspect="auto")
            ax.set_title(f"mask {i}")
            ax.set_xticks(range(len(self.feature_names)))
            ax.set_xticklabels(self.feature_names, rotation=45)
            ax.set_xlabel("Features")
            ax.set_ylabel("Samples")

        for j in range(n_masks, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            axs[row][col].axis("off")

        plt.suptitle(f"{self.msg_name}(selected) TabNet Masks", fontsize=30, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.mask_dir, f"{self.msg_name}(selected)_masks.png"))
        plt.close()
        

    def plot_feature_importance(self):
        mean_feature_importance = np.mean(self.explain_matrix, axis=0)
        sorted_idx = np.argsort(mean_feature_importance)[::-1]
        sorted_importance = mean_feature_importance[sorted_idx]
        sorted_features = [self.feature_names[i] for i in sorted_idx]

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # 격자를 막대 뒤에 그리기
        ax.set_axisbelow(True)
        ax.barh(range(len(sorted_importance)), sorted_importance[::-1], color='cornflowerblue')

        # y축 설정
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features[::-1])

        # 라벨 & 제목
        ax.set_ylabel("Feature")
        ax.set_xlabel("Importance")
        ax.set_title(f"{self.msg_name}(selected) TabNet Feature Importance", fontsize=16, fontweight="bold", pad=10)

        # grid 설정: x축에 점선으로, 막대 뒤에 위치
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.imp_dir, f"{self.msg_name}(selected)_sorted_importance.png"))
        plt.close()

    def plot_loss(self):
        loss_df = pd.DataFrame({
            "train_loss": self.history["loss"],
            "valid_loss": self.history["valid_logloss"]
        })

        loss_df.to_csv(os.path.join(self.loss_dir, f"{self.msg_name}_loss_history.csv"), index=False)
        # print(f"Loss 기록 저장 완료")

        # Best Epoch 계산
        best_epoch = np.argmin(self.history['valid_logloss']) + 1

        # 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(loss_df['train_loss'], label="Train Loss")
        plt.plot(loss_df['valid_loss'], label="Validation Loss")
        plt.axvline(x=best_epoch-1, color='red', linestyle='--', label=f'Best Epoch {best_epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        title_font = {
            "fontsize": 16,
            "fontweight": "bold"
        }
        plt.title(f'{self.msg_name}(selected) Loss', fontdict=title_font, pad=10)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.loss_dir, f"{self.msg_name}(selected)_loss_curve.png"))
        # plt.show()

    def plot_all(self):
        """모든 결과 시각화: 마스크, 중요도, 손실 곡선"""
        self.plot_confusion_matrix()
        self.plot_masks()
        self.plot_feature_importance()
        self.plot_loss()
        print(f"[{self.msg_name}] 시각화 저장 완료")


def train_visualize(data_path, save_dir=save_dir):
    pipeline = TabNetPipeline(data_path=data_path, scaler=StandardScaler(), save_dir=save_dir)
    pipeline.load_prepare_data()
    pipeline.train()

    results = pipeline.val_evaluate(save_csv=True)

    visualizer = ResultsVisualizer(pipeline)
    visualizer.plot_all()

    return results

import traceback

# 선정된 피처 파일 불러오기
feature_df = pd.read_csv('../0.data/selected_features_20250529.csv')
msg_lst = list(feature_df['msg_field'].values)

if __name__ == "__main__":
    error_log_path = os.path.join(save_dir, "error_log.txt")

    for msg_name in msg_lst:
        path = f"../0.data/results/{msg_name}_merged.csv"
        print(f"\n Processing file: {os.path.basename(path)}")

        try:
            train_visualize(data_path=path)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)} 처리 중 오류 발생: {e}")
            with open(error_log_path, "a") as f:
                f.write(f"File: {path}\n")
                f.write(f"{traceback.format_exc()}\n\n")
