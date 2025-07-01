import numpy as np
import onnx
from onnx.tools import net_drawer
import pandas as pd
from sklearn.model_selection import train_test_split

file = pd.read_csv('./gripon_csv/merged_output/AHR2_merged.csv')
n_file = file[file['label']==0]
y_n_file = n_file[['label']]
ab_file = file[file['label']==1]
y_ab_file = ab_file[['label']]
n_file.drop(columns='label',inplace=True)
ab_file.drop(columns='label',inplace=True)
# y_n_file['loginfo']=0
# y_ab_file['loginfo']=1
# y_n_file.columns = y_n_file.columns.str.replace('loginfo','label')
# y_ab_file.columns = y_ab_file.columns.str.replace('loginfo','label')
tr_n_file,tt_n_file,tr_y_n_file,tt_y_n_file=train_test_split(n_file,y_n_file,test_size=0.1,random_state=42,shuffle=True)
tr_ab_file,tt_ab_file,tr_y_ab_file,tt_y_ab_file=train_test_split(ab_file,y_ab_file,test_size=0.1,random_state=42,shuffle=True)

tr_n_data = pd.concat([tr_n_file.reset_index(drop=True),tr_y_n_file.reset_index(drop=True)],axis=1)
tr_ab_data = pd.concat([tr_ab_file.reset_index(drop=True),tr_y_ab_file.reset_index(drop=True)],axis=1)
tt_n_data = pd.concat([tt_n_file.reset_index(drop=True),tt_y_n_file.reset_index(drop=True)],axis=1)
tt_ab_data = pd.concat([tt_ab_file.reset_index(drop=True),tt_y_ab_file.reset_index(drop=True)],axis=1)

tr_data = pd.concat([tr_n_data.reset_index(drop=True), tr_ab_data.reset_index(drop=True)],axis=0)
tt_data = pd.concat([tt_n_data.reset_index(drop=True), tt_ab_data.reset_index(drop=True)],axis=0)

# tr_data.to_csv('gripon_train_AHR2.csv',index=False)
cal_data = tr_data[['Roll','Pitch','Yaw']]
# tt_data.to_csv('gripon_test_AHR2.csv',index=False)
cal_data.to_numpy()
np.save('./test/AHR2_SAINT_best_model.npy',cal_data)

# cal_data = np.load('./calib_data/XKF1+2_calib.npy')
# print(cal_data.shape)
# random_indices = np.random.choice(cal_data.shape[0], size=10, replace=False)
# random_samples = cal_data[random_indices]
# print(random_samples)
# reshaped_samples = random_samples.reshape(10, 1, 1, 8).astype(np.float32)
# print(reshaped_samples[1].shape)

# import onnx
# from onnxsim import simplify

# model = onnx.load("ft_transformer_model.onnx")
# model_simp, check = simplify(model)
# onnx.save(model_simp, "ft_transformer_model_simplified.onnx")
