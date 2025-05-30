import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from pycaret.classification import setup, compare_models, pull
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("SHAP 계산 시간 초과")


class SHAPPipeline:
    def __init__(self, data_path, msg_name, save_dir=None,):
        self.data_path = data_path
        self.msg_name = msg_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.shap_csv_path = os.path.join(self.save_dir, "all_fields_shap_importance_ratio.csv")
        self.model_csv_path = os.path.join(self.save_dir, "all_fields_model_comparison.csv")

        # 선정된 피처 목록 파일 불러오기
        self.feature_df = pd.read_csv("../0.data/selected_features_20250529.csv")

    def run(self):
        try:
            print(f"\n▶ Processing {self.msg_name}...")

            # 선정된 피처 리스트
            feature_lst = self.feature_df.loc[self.feature_df["msg_field"] == "AHR2", "feature_list"].values[0].split(", ")
            feature_lst.append("label")

            # 데이터 로드 및 전처리
            df = pd.read_csv(self.data_path)
            n_df = df.drop(["timestamp", "TimeUS"], axis=1, errors="ignore")
            n_df = n_df[feature_lst]

            setup(data=n_df, target='label', session_id=42, verbose=False)
            tree_models = ['rf', 'dt', 'gbc']
            best_model = compare_models(include=tree_models, sort="Recall")
            
            # 모델 이름 추출
            model_name = best_model.__class__.__name__
            print(f"✔ 선택된 모델: {model_name}")

            # 모델 성능 저장
            model_result = pull()
            model_result['field'] = self.msg_name
            model_result = model_result[['field', 'Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)']]
            self.append_to_csv(model_result, self.model_csv_path)

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
            shap_importance['field'] = self.msg_name
            shap_importance['model'] = model_name
            shap_importance['shap_ratio'] = shap_importance['mean_abs_shap'] / shap_importance['mean_abs_shap'].sum()
            shap_importance = shap_importance[['field', 'model', 'feature', 'mean_abs_shap', 'shap_ratio']]
            shap_importance = shap_importance.sort_values(by='mean_abs_shap', ascending=False)

            self.append_to_csv(shap_importance, self.shap_csv_path)
            self.plot_shap_bar(shap_importance, model_name)

            print(f"✔ {self.msg_name} 처리 완료")


        except Exception as e:
            print(f"XXXXX {self.msg_name} 처리 중 오류 발생: {e} XXXXX")

    def append_to_csv(self, df, csv_path):
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', index=False, header=False)
        else:
            df.to_csv(csv_path, index=False)

    def plot_shap_bar(self, shap_importance, model_name, top_n=10):
        top_features = shap_importance.sort_values(by='mean_abs_shap', ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(top_features['feature'][::-1], top_features['mean_abs_shap'][::-1], color='royalblue')
        plt.title(f"{self.msg_name} Top {top_n} SHAP Features\n(Best: {model_name})", fontsize=14, fontweight="bold", pad=10)
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Features')

        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_save_path = os.path.join(self.save_dir, f"{self.msg_name}_shap_barplot_{model_name}.png")
        plt.savefig(plot_save_path, dpi=300)
        plt.close()
        print(f"   -> SHAP 바 플롯 저장 완료: {plot_save_path}")



import os
import glob
import traceback
import signal

# 경로 설정
save_dir = "./results/ml_shap"
os.makedirs(save_dir, exist_ok=True)

# 처리 대상 파일 목록
feature_df = pd.read_csv('../0.data/selected_features_20250529.csv')
msg_lst = list(feature_df['msg_field'].values)

# 에러 로그 파일 경로
error_log_path = os.path.join(save_dir, "error_log.txt")

# SHAPPipeline 클래스 임포트 or 같은 파일에 정의되어 있어야 함
# from shap_pipeline import SHAPPipeline

if __name__ == "__main__":
    for msg_name in msg_lst:
        path = f"../0.data/results/{msg_name}_merged.csv"
        print(f"\n▶ Processing file: {os.path.basename(path)}")

        try:
            pipeline = SHAPPipeline(data_path=path,msg_name=msg_name, save_dir=save_dir)
            pipeline.run()
        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)} 처리 중 오류 발생: {e}")
            with open(error_log_path, "a") as f:
                f.write(f"File: {path}\n")
                f.write(traceback.format_exc())
                f.write("\n\n")
