import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import umap.umap_ as umap
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

class DimReducer:
    def __init__(self, filepath, label_col="label"):
        self.filepath = filepath
        self.label_col = label_col
        self.msg_field = os.path.basename(filepath).split("_")[0]
        self.df = None
        self.y = None
        self.features = None

    def load(self):
        df = pd.read_csv(self.filepath)
        df = df.drop(columns=["timestamp", "TimeUS"], errors="ignore")
        df = df.dropna(axis=1, how='all')
        self.df = df
        self.y = df[self.label_col]
        self.features = [col for col in df.columns if col != self.label_col]
        self.n_samples = len(df)

    def _sample_and_scale(self, random_state=42, sample_size=3000):
        sample_size = min(sample_size, self.n_samples)
        df_sample, _ = train_test_split(
            self.df, train_size=sample_size, stratify=self.df[self.label_col], random_state=random_state
        )
        X = df_sample[self.features]
        y = df_sample[self.label_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def run_pca(self, save_dir="analyze/pca"):
        n_repeat = 4 if self.n_samples >= 3000 else 1
        os.makedirs(save_dir, exist_ok=True)
        rows = math.ceil(n_repeat / 2)
        cols = 2 if n_repeat > 1 else 1
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axs = np.array(axs).flatten() if n_repeat > 1 else [axs]
        all_components = []

        for i in range(n_repeat):
            X_scaled, y = self._sample_and_scale(random_state=42 + i)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pc1 = pca.components_[0]
            anchor_index = abs(pc1).argmax()
            pca.components_ *= 1 if pc1[anchor_index] >= 0 else -1

            colors = y.map({0: "green", 1: "red"})
            axs[i].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5)
            axs[i].set_title(f"PCA Sample {i+1}")
            axs[i].set_xlabel("PC1")
            axs[i].set_ylabel("PC2")
            axs[i].grid(True)

            comp_df = pd.DataFrame(pca.components_.T, index=self.features, columns=["PC1", "PC2"])
            comp_df["feature"] = comp_df.index
            comp_df["sample_id"] = i + 1
            comp_df["msg_field"] = self.msg_field
            all_components.append(comp_df[["msg_field", "sample_id", "feature", "PC1", "PC2"]])

        plt.suptitle(f"{self.msg_field} PCA", fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{self.msg_field}_pca_repeat.png"), dpi=300)
        plt.close()

        final_df = pd.concat(all_components, ignore_index=True)
        final_df.to_csv(os.path.join(save_dir, f"{self.msg_field}_pca_components.csv"), index=False)

        summary_df = final_df.groupby("feature")[["PC1", "PC2"]].agg(["mean", "std"])

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        for j, pc in enumerate(["PC1", "PC2"]):
            means = summary_df[(pc, "mean")]
            sorted_idx = means.abs().sort_values(ascending=False).index
            means = means.loc[sorted_idx]
            bars = axs[j].bar(means.index, means.values, color="skyblue")
            axs[j].axhline(0, color="gray", linestyle="--", linewidth=0.8)
            axs[j].set_title(f"{pc} Loadings (Sorted by |Mean|)", fontsize=13, fontweight='bold')
            axs[j].tick_params(axis='x', rotation=45)
            axs[j].grid(axis='y', linestyle=':', linewidth=0.5)

        plt.suptitle(f"{self.msg_field} PCA Loading Summary", fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{self.msg_field}_pca_summary_sorted.png"), dpi=300)
        plt.close()
        print(f"{self.msg_field} [PCA 완료 + 정렬 + 요약 그래프 저장]")

    def run_umap(self, save_dir="analyze/umap"):
        n_repeat = 4 if self.n_samples >= 3000 else 1
        os.makedirs(save_dir, exist_ok=True)
        rows = math.ceil(n_repeat / 2)
        cols = 2 if n_repeat > 1 else 1
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axs = np.array(axs).flatten() if n_repeat > 1 else [axs]

        for i in range(n_repeat):
            X_scaled, y = self._sample_and_scale(random_state=42 + i)
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            colors = y.map({0: "green", 1: "red"})
            axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=colors, alpha=0.5)
            axs[i].set_title(f"UMAP Sample {i+1}")
            axs[i].set_xlabel("UMAP1")
            axs[i].set_ylabel("UMAP2")
            axs[i].grid(True)

        plt.suptitle(f"{self.msg_field} UMAP", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{self.msg_field}_umap_repeat.png"), dpi=300)
        plt.close()
        print(f"{self.msg_field} [UMAP 완료]")

    def run_all(self):
        self.load()
        self.run_pca()
        self.run_umap()

if __name__ == "__main__":
    
    paths = sorted(glob.glob("../data/results/*.csv"))

    for path in paths:
        lst = ["MAVC", "PM", "SRTL", "TSYN", "XKT"]

        if os.path.basename(path).split("_")[0] in lst:
            print(f"Processing file: {os.path.basename(path)}")

            try: 
                dimreducer = DimReducer(path)
                dimreducer.run_all()

            except Exception as e:
                    print(f"[ERROR] {os.path.basename(path)} 처리 중 오류 발생: {e}")
        
        else:
            continue


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import glob

# import umap.umap_ as umap


# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# matplotlib.rcParams['font.family'] = 'NanumGothic'
# matplotlib.rcParams['axes.unicode_minus'] = False

# class DimReducer:
#     def __init__(self, filepath, label_col="label"):
#         self.filepath = filepath
#         self.label_col = label_col
#         self.msg_field = os.path.basename(filepath).split("_")[0]
#         self.df = None
#         self.y = None
#         self.features = None

#     def load(self):
#         df = pd.read_csv(self.filepath)
#         df = df.drop(columns=["timestamp", "TimeUS"], errors="ignore")
#         df = df.dropna(axis=1, how='all')
#         self.df = df
#         self.y = df[self.label_col]
#         self.features = [col for col in df.columns if col != self.label_col]
#         self.n_samples = len(df)  # 샘플 수 저장


#     def _sample_and_scale(self, random_state=42, sample_size=3000):
#         df_sample, _ = train_test_split(
#             self.df, train_size=sample_size, stratify=self.df[self.label_col], random_state=random_state
#         )
#         X = df_sample[self.features]
#         y = df_sample[self.label_col]
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#         return X_scaled, y
    
#     def run_pca(self, save_dir="analyze/pca", n_repeat=4):
#         n_repeat = 4 if self.n_samples >= 3000 else 1
#         os.makedirs(save_dir, exist_ok=True)
#         fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#         axs = axs.flatten()
#         all_components = []

#         for i in range(n_repeat):
#             X_scaled, y = self._sample_and_scale(random_state=42 + i)
#             pca = PCA(n_components=2)
#             X_pca = pca.fit_transform(X_scaled)

#             # Sign alignment 기준: PC1의 절댓값 최대 feature
#             pc1 = pca.components_[0]
#             anchor_index = abs(pc1).argmax()
#             anchor_sign = 1 if pc1[anchor_index] >= 0 else -1
#             pca.components_ *= anchor_sign

#             # 시각화
#             colors = y.map({0: "green", 1: "red"})
#             axs[i].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5)
#             axs[i].set_title(f"PCA Sample {i+1}")
#             axs[i].set_xlabel("PC1")
#             axs[i].set_ylabel("PC2")
#             axs[i].grid(True)

#             # 주성분 로딩 저장
#             comp_df = pd.DataFrame(pca.components_.T, index=self.features, columns=["PC1", "PC2"])
#             comp_df["feature"] = comp_df.index
#             comp_df["sample_id"] = i + 1
#             comp_df["msg_field"] = self.msg_field
#             all_components.append(comp_df[["msg_field", "sample_id", "feature", "PC1", "PC2"]])

#         # 시각화 저장
#         plt.suptitle(f"{self.msg_field} PCA", fontsize=20, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f"{self.msg_field}_pca_repeat.png"), dpi=300)
#         plt.close()

#         # CSV 저장
#         final_df = pd.concat(all_components, ignore_index=True)
#         final_df.to_csv(os.path.join(save_dir, f"{self.msg_field}_pca_components.csv"), index=False)

#         # 평균 ± 표준편차 요약 그래프
#         summary_df = final_df.groupby("feature")[["PC1", "PC2"]].agg(["mean", "std"])

#         fig, axs = plt.subplots(1, 2, figsize=(14, 5))

#         for j, pc in enumerate(["PC1", "PC2"]):
#             means = summary_df[(pc, "mean")]
            
#             # 절댓값 기준으로 정렬
#             sorted_idx = means.abs().sort_values(ascending=False).index
#             means = means.loc[sorted_idx]

#             bars = axs[j].bar(
#                 means.index,
#                 means.values,
#                 color="skyblue",
#             )

#             # 기준선
#             axs[j].axhline(0, color="gray", linestyle="--", linewidth=0.8)
#             axs[j].set_title(f"{pc} Loadings (Sorted by |Mean|)", fontsize=13, fontweight='bold')
#             axs[j].tick_params(axis='x', rotation=45)
#             axs[j].grid(axis='y', linestyle=':', linewidth=0.5)

#         plt.suptitle(f"{self.msg_field} PCA Loading Summary", fontsize=20, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f"{self.msg_field}_pca_summary_sorted.png"), dpi=300)
#         plt.close()

#         print(f"{self.msg_field} [PCA 완료 + 정렬 + 요약 그래프 저장]")

#     def run_umap(self, save_dir="analyze/umap", n_repeat=4):
#         n_repeat = 4 if self.n_samples >= 3000 else 1
#         os.makedirs(save_dir, exist_ok=True)
#         fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#         axs = axs.flatten()

#         for i in range(n_repeat):
#             X_scaled, y = self._sample_and_scale(random_state=42 + i)
#             reducer = umap.UMAP(n_components=2, random_state=42)
#             X_umap = reducer.fit_transform(X_scaled)

#             colors = y.map({0: "green", 1: "red"})
#             axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=colors, alpha=0.5)
#             axs[i].set_title(f"UMAP Sample {i+1}")
#             axs[i].set_xlabel("UMAP1")
#             axs[i].set_ylabel("UMAP2")
#             axs[i].grid(True)

#         plt.suptitle(f"{self.msg_field} UMAP", fontsize=20, fontweight="bold")
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f"{self.msg_field}_umap_repeat.png"), dpi=300)
#         plt.close()
#         print(f"{self.msg_field} [UMAP 완료]")

#     def run_all(self):
#         self.load()
#         self.run_pca()
#         self.run_umap()

