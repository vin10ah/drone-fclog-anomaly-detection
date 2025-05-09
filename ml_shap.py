import numpy as np
import pandas as pd
import os
import glob
import shap
import warnings
import matplotlib.pyplot as plt
from pycaret.classification import setup, compare_models, pull
import matplotlib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']


class SHAPPipeline:
    def __init__(self, data_dir="../data/results/", save_dir="shap_results/plots"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.msg_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        self.all_shap_results = []
        self.all_model_results = []
        print(f"Found {len(self.msg_paths)} files.")

    def process_file(self, filepath):
        msg_name = os.path.basename(filepath).split('_')[0]
        print(f"\n▶ Processing {msg_name}...")

        try:
            # 데이터 로드
            df = pd.read_csv(filepath)
            n_df = df.drop(["timestamp", "TimeUS"], axis=1, errors="ignore")

            # PyCaret 모델링
            setup(data=n_df, target='label', session_id=42, verbose=False)
            tree_models = ['rf', 'et', 'dt', 'ada', 'gbc']
            best_model = compare_models(include=tree_models, sort="Recall")

            # 모델 성능 저장
            model_result = pull()
            model_result['field'] = msg_name
            model_result = model_result[['field', 'Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)']]
            self.all_model_results.append(model_result)

            # 클래스 균형 샘플링
            normal = n_df[n_df['label'] == 0]
            anomaly = n_df[n_df['label'] == 1]

            normal_sample = normal.sample(n=min(len(normal), 1500), random_state=42)
            anomaly_sample = anomaly.sample(n=min(len(anomaly), 1500), random_state=42)

            sample_df = pd.concat([normal_sample, anomaly_sample])
            X = sample_df.drop(columns=['label'])

            # SHAP 계산
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_array = shap_values[:, :, 1]
            elif isinstance(shap_values, list):
                shap_array = np.array(shap_values[1])
            else:
                shap_array = np.array(shap_values)

            shap_importance = pd.DataFrame({
                'feature': X.columns,
                'mean_abs_shap': np.abs(shap_array).mean(axis=0)
            })
            shap_importance['field'] = msg_name
            shap_importance['shap_ratio'] = shap_importance['mean_abs_shap'] / shap_importance['mean_abs_shap'].sum()
            shap_importance = shap_importance[['field', 'feature', 'shap_ratio']]
            shap_importance = shap_importance.sort_values(by='shap_ratio', ascending=False)

            self.all_shap_results.append(shap_importance)

            # 개별 shap barplot 바로 저장
            self.plot_shap_bar(shap_importance, msg_name)

        except Exception as e:
            print(f"XXXXX {msg_name} 분석 실패: {e} XXXXX")

    def run_all(self):
        for path in self.msg_paths:
            self.process_file(path)

    def save_results(self):
        # SHAP 중요도 통합 저장
        shap_df = pd.concat(self.all_shap_results, ignore_index=True)
        shap_save_path = os.path.join(self.save_dir, "all_fields_shap_importance_ratio.csv")
        shap_df.to_csv(shap_save_path, index=False)
        print(f"\n✔ SHAP 중요도 저장 완료: {shap_save_path}")

        # 모델 비교 통합 저장
        model_df = pd.concat(self.all_model_results, ignore_index=True)
        model_save_path = os.path.join(self.save_dir, "all_fields_model_comparison.csv")
        model_df.to_csv(model_save_path, index=False)
        print(f"✔ 모델 성능 저장 완료: {model_save_path}")

    def plot_shap_bar(self, shap_importance, field_name, top_n=10):
        top_features = shap_importance.head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(top_features['feature'][::-1], top_features['shap_ratio'][::-1], color='skyblue')
        plt.title(f"Top {top_n} SHAP Features - {field_name}", fontsize=16)
        plt.xlabel('SHAP Importance Ratio')
        plt.ylabel('Features')
        plt.grid(True)
        plt.tight_layout()

        plot_save_path = os.path.join(self.save_dir, f"{field_name}_shap_barplot.png")
        plt.savefig(plot_save_path, dpi=300)
        plt.close()
        print(f"   -> SHAP 바 플롯 저장 완료: {plot_save_path}")


if __name__ == "__main__":
    pipeline = SHAPPipeline()
    pipeline.run_all()
    pipeline.save_results()