import os
import pandas as pd

desktop = r"C:\Users\shari\Desktop"
output_file = r"e:\Ai Phishing Detector\scripts\datasets_summary.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for i in range(1, 7):
        fn = f"dataset{i}.csv"
        fp = os.path.join(desktop, fn)
        if os.path.exists(fp):
            out.write(f"=== {fn} ===\n")
            try:
                df = pd.read_csv(fp, nrows=3)
                out.write(f"Cols: {df.columns.tolist()[:15]}\n")
                out.write(f"Head:\n{df.head(1).to_string()}\n")
            except Exception as e:
                out.write(f"Error reading: {e}\n")
            out.write("\n")
