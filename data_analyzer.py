import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import matplotlib.ticker as ticker 
import glob


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

    
    def plot_feature_trend(self, feature: str, ax=None, sample_rate=10):
        if self.df is None:
            raise ValueError("Data is not loaded. Please run load(filepath) first.")

        # 정상/이상 데이터 분리
        normal = self.df[self.df[self.label_col] == 0]
        abnormal = self.df[self.df[self.label_col] == 1]

        # 샘플링
        normal = normal.iloc[::sample_rate]
        abnormal = abnormal.iloc[::sample_rate]

        # x축: 각각 길이 맞춰 0부터
        x_normal = range(len(normal))
        x_abnormal = range(len(abnormal))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # 정상 plot
        ax.plot(x_normal, normal[feature], color='blue', alpha=0.7, linewidth=0.7, label='Normal')

        # 이상 plot
        ax.plot(x_abnormal, abnormal[feature], color='red', alpha=0.8, linewidth=0.7, label='Abnormal')

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.set_title(f'{feature}', fontsize=16)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'{feature} Value')
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            # plt.show()



    def plot(self, save_dir="plots"):
        if self.df is None:
            raise ValueError("Data is not loaded. Please run load(filepath) first.")

        features = [col for col in self.df.columns if col != self.label_col]
        if not features:
            print("No features to plot.")
            return

        n = len(features)
        if n <= 4:
            ncols = n
            nrows = 1

        else:
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

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig)


    def get_summary_stats(self):
        if self.df is None:
            raise ValueError("Data is not loaded. Please run load(filepath) first.")

        row_count, col_count = self.df.shape
        column_names = list(self.df.columns)
        missing = self.df.isnull().any().any()

        stats = self.df.describe().T[['min', 'max', 'mean']]

        summary = []
        for col in stats.index:
            if col == self.label_col:
                continue  # 레이블 제외

            col_data = self.df[col].dropna()

            summary.append({
                "msg_field": self.msg_field,
                "feature": col,
                "rows": row_count,
                "cols": col_count,
                "missing": missing,
                "min": stats.loc[col, "min"],
                "max": stats.loc[col, "max"],
                "mean": stats.loc[col, "mean"],
                "var": col_data.var(),
                "non_zero_rate": (col_data != 0).mean(),
                "unique_count": col_data.nunique(),
            })
        return summary



if __name__ == "__main__":
    paths = sorted(glob.glob("/home/seobin1027/tasks/new_log_data/data/results/*.csv"))
    save_dir = "plots"

    summary_stats = []

    for path in paths:
        print(f"\nProcessing file: {os.path.basename(path)}")

        try:
            analyzer = DataAnalyzer(label_col='label')
            analyzer.load(path)
            analyzer.plot(save_dir=save_dir)

            summary_stats.extend(analyzer.get_summary_stats())  # 통계 수집

        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)} 처리 중 오류 발생: {e}")

    # 최종 통계 CSV 저장
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv("log_inform_summary.csv", index=False)
    print("log_summary.csv 저장 완료")
