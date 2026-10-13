import os
import pandas as pd

desktop = r"C:\Users\shari\Desktop"
output_file = r"e:\Ai Phishing Detector\scripts\datasets_detail.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for name in ["dataset3.csv", "dataset4.csv", "dataset5.csv"]:
        fp = os.path.join(desktop, name)
        if os.path.exists(fp):
            out.write(f"=== {name} ===\n")
            try:
                df = pd.read_csv(fp, low_memory=False, encoding="utf-8")
            except Exception:
                df = pd.read_csv(fp, low_memory=False, encoding="latin-1")
            out.write(f"Rows: {len(df):,}, Cols: {len(df.columns)}\n")
            # find label col
            for col in df.columns:
                if col.lower() in ["label", "status", "class", "result", "type"]:
                    out.write(f"Label '{col}' counts:\n{df[col].value_counts(dropna=False)}\n")
            # check domain/url col
            for col in df.columns:
                if col.lower() in ["url", "domain"]:
                    out.write(f"Sample '{col}':\n{df[col].dropna().head(3)}\n")
            out.write("\n")
