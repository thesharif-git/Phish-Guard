import os

desktop = r"C:\Users\shari\Desktop"
output_file = r"e:\Ai Phishing Detector\scripts\desktop_files.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Desktop Datasets / Files:\n")
    try:
        files = os.listdir(desktop)
        for fn in sorted(files):
            if fn.endswith((".csv", ".parquet", ".json", ".txt", ".xlsx")):
                full = os.path.join(desktop, fn)
                size_mb = os.path.getsize(full) / (1024 * 1024)
                f.write(f"{fn} ({size_mb:.2f} MB)\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
