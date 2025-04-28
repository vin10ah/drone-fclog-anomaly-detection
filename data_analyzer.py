import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime


class DataAnalyzer:
    def __init__(self, label_col='label'):
        self.df = None
        self.label_col = label_col
        self.msg_field = None

    def load(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.df = self.df.drop(columns=['timestamp', 'TimeUS'], errors='ignore')
        self.msg_field = os.path.basename(filepath).split("_")[0]

        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]

        if missing_cols.empty:
            missing_info = "No missing columns"
        else:
            missing_info = f"Missing columns ({len(missing_cols)}): {list(missing_cols.index)}"

        print(f"Message field: {self.msg_field} | Shape: {self.df.shape} | {missing_info}")

    def plot_feature_trend(self, feature: str, ax=None):
        if self.df is None:
            raise ValueError("Data is not loaded. Please run load(filepath) first.")

        normal = self.df[self.df[self.label_col] == 0]
        abnormal = self.df[self.df[self.label_col] == 1]

        if ax is None:
            plt.plot(normal.index, normal[feature], color='blue', label='Normal')
            plt.plot(abnormal.index, abnormal[feature], color='red', label='Abnormal')
            plt.title(f'Feature Value Trend by Sequence : {feature}')
            plt.xlabel('Index')
            plt.ylabel(f'{feature} Value')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            ax.plot(normal.index, normal[feature], color='blue', label='Normal')
            ax.plot(abnormal.index, abnormal[feature], color='red', label='Abnormal')
            ax.set_title(f'{feature}')
            ax.set_xlabel('Index')
            ax.set_ylabel(f'{feature} Value')
            ax.grid(True)

    def plot(self, save_dir="plots"):
        if self.df is None:
            raise ValueError("Data is not loaded. Please run load(filepath) first.")

        features = [col for col in self.df.columns if col != self.label_col]
        if not features:
            print("No features to plot.")
            return

        n = len(features)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            self.plot_feature_trend(feature, ax=axes[i])

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'Distribution of Each Feature - {self.msg_field}', 
                     fontsize=24,
                     y=1.02
                     )
        plt.tight_layout()

        # 저장 디렉토리 없으면 생성
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{self.msg_field}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close(fig)
















# def load_data(filepath: str):
#     df = pd.read_csv(filepath)
#     df = df.drop(columns=['timestamp', 'TimeUS'], errors='ignore')
#     return df

# class Preprocessor:
#     def __init__(self, filepath: str):
#         self.name = os.path.basename(filepath).split("_")[0]
#         self.df = load_data(filepath)
#         self.preprocess()

#     def preprocess(self):
#         missing = self.df.isnull().sum()
#         missing_cols = missing[missing > 0]

#         if missing_cols.empty:
#             missing_info = "No missing columns"
#         else:
#             missing_info = f"Missing columns ({len(missing_cols)}): {list(missing_cols.index)}"

#         print(f"Message field: {self.name} | Shape: {self.df.shape} | {missing_info}")

#     def to_analyzer(self, label_col='label'):
#         return DataAnalyzer(self.df, label_col=label_col, msg_field=self.name)



# class DataAnalyzer:
#     def __init__(self, df, label_col='label', msg_field):
#         self.df = df
#         self.label_col = label_col
#         self.msg_field = msg_field

#     def plot_feature_trend(self, feature: str, ax=None):
#         normal = self.df[self.df[self.label_col] == 0][feature]
#         abnormal = self.df[self.df[self.label_col] == 1][feature]

#         if ax is None:
#             plt.plot(normal.index, normal[feature], color='blue', label='Normal')
#             plt.plot(abnormal.index, abnormal[feature], color='red', label='Abnormal')
#             plt.title(f'Feature Value Trend by Sequence : {feature}')
#             plt.xlabel('Index')
#             plt.ylabel(f'{feature} Value')
#             plt.legend()
#             plt.grid(True)
#             plt.show()
        
#         else:
#             ax.plot(normal.index, normal[feature], color='blue', label='Normal')
#             ax.plot(abnormal.index, abnormal[feature], color='red', label='Abnormal')
#             ax.set_title(f'{feature}')
#             ax.set_xlabel('Index')
#             ax.set_ylabel(f'{feature} Value')
#             ax.grid(True)

#     def plot_fields(self, bins=50):
#         features = [col for col in self.df.columns if col != self.label_col]

#         if not features:
#             print("No features to plot.")
#             return
        
#         n = len(features)
#         if n <= 4:
#             ncols = n
#             nrows = 1

#         else:
#             ncols = 4
#             nrows = (n + 1) // ncols

#         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
#         axes = axes.flatten()

#         for i, feature in enumerate(features):
#             self.plot_feature_trend(feature, ax=axes[i])

#         for j in range(i + 1, len(axes)):
#             fig.delaxes(axes[j])  # 남는 빈 plot 삭제

#         fig.suptitle(f'Distribution of Each Feature {self.msg_field}', fontsize=16)
#         plt.tight_layout()
#         plt.show()
