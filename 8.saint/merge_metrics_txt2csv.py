import os
import glob
import pandas as pd

def merge_metrics_txt_to_csv(metrics_dir, output_csv_path):
    records = []

    for txt_file in glob.glob(os.path.join(metrics_dir, "*_metrics.txt")):
        msg_field = os.path.basename(txt_file).replace("_metrics.txt", "")

        record = {"msg_field": msg_field}

        with open(txt_file, "r") as f:
            for line in f:
                if ':' not in line:
                    continue
                key, val = line.strip().split(': ')
                if key == "best_epoch":
                    record[key] = int(val)
                else:
                    record[key] = float(val)

        records.append(record)

    df = pd.DataFrame(records)
    df.sort_values(by="msg_field", inplace=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved summary CSV to: {output_csv_path}")


if __name__ == "__main__":
    metrics_folder = "./results/metrics"
    output_csv = "./results/SAINT_summary.csv"
    merge_metrics_txt_to_csv(metrics_folder, output_csv)
