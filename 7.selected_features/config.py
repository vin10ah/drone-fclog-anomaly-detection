import glob
import os


version = "add_error"
file_paths = sorted(glob.glob("../0.data/merged_add_error/*.csv"))
selected_path = '../0.data/selected_features_20250602.csv'
msg_lst = [os.path.basename(path).split("_")[0] for path in file_paths]
