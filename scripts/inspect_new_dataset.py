import os
import sys
import pandas as pd

dataset_path = r"C:\Users\shari\Desktop\dataset_cybersecurity_michelle.csv"
output_path = r"e:\Ai Phishing Detector\scripts\dataset_michelle_summary.txt"

def analyze():
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Analyzing: {dataset_path}\n")
        if not os.path.exists(dataset_path):
            out.write("File does not exist!\n")
            return
        
        file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        out.write(f"File size: {file_size_mb:.2f} MB\n\n")

        # Read first 100 rows to quickly get schema
        preview_df = pd.read_csv(dataset_path, nrows=5)
        out.write(f"Columns ({len(preview_df.columns)}):\n")
        for col in preview_df.columns:
            out.write(f" - {col}\n")
        out.write("\nSample head (first 2 rows):\n")
        out.write(preview_df.head(2).to_string() + "\n\n")

        # Read full file
        full_df = pd.read_csv(dataset_path, low_memory=False)
        out.write(f"Total rows: {len(full_df):,}\n")
        out.write(f"Total columns: {len(full_df.columns)}\n\n")

        # Check label or target candidates
        label_cols = [c for c in full_df.columns if any(k in c.lower() for k in ['label', 'class', 'type', 'target', 'phish', 'result', 'status', 'category'])]
        out.write(f"Candidate label columns: {label_cols}\n\n")
        for lc in label_cols:
            out.write(f"Value counts for '{lc}':\n")
            out.write(full_df[lc].value_counts(dropna=False).to_string() + "\n\n")

        # Check domain/URL candidates
        url_cols = [c for c in full_df.columns if any(k in c.lower() for k in ['url', 'domain', 'host', 'site', 'fqdn', 'link', 'web'])]
        out.write(f"Candidate URL/Domain columns: {url_cols}\n\n")
        for uc in url_cols:
            out.write(f"Sample values for '{uc}':\n")
            out.write(full_df[uc].dropna().head(5).to_string() + "\n\n")

        # Check missing values
        out.write("Missing values count:\n")
        out.write(full_df.isnull().sum().to_string() + "\n")

if __name__ == "__main__":
    analyze()
