import pandas as pd
import numpy as np
import re
import os
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb


# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import clean_text, extract_features



# ─────────────────────────────────────────
#  Main Training Function
# ─────────────────────────────────────────
def train_model():
    print("=" * 60)
    print("  PhishGuard AI — LightGBM Model Training")
    print("=" * 60)

    desktop_path = r"C:\Users\shari\OneDrive\Desktop"

    dataset_configs = [
        {"file": "Phishing_Email.csv",      "text_col": "Email Text",    "label_col": "Email Type"},
        {"file": "All phishing_mails.csv",  "text_col": "text_combined", "label_col": "label"},
        {"file": "CEAS_08.csv",             "text_col": "body",          "label_col": "label"},
        {"file": "Enron.csv",               "text_col": "body",          "label_col": "label"},
        {"file": "Nigerian_Fraud.csv",      "text_col": "body",          "label_col": "label"},
        {"file": "SpamAssasin.csv",         "text_col": "body",          "label_col": "label"},
    ]

    dataframes = []

    for config in dataset_configs:
        file_path = os.path.join(desktop_path, config["file"])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Not found: {config['file']}")
            continue

        print(f"  [LOAD] {config['file']}...")
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')

            cols_to_use = [config['text_col'], config['label_col']]
            if config["file"] == "Enron.csv" and "subject" in df.columns:
                cols_to_use.append("subject")

            existing_cols = [c for c in cols_to_use if c in df.columns]
            df = df[existing_cols]

            if config["file"] == "Enron.csv" and "subject" in df.columns:
                df[config['text_col']] = df["subject"].fillna('') + " " + df[config['text_col']].fillna('')

            df = df.rename(columns={config['text_col']: 'text', config['label_col']: 'label'})
            df = df[['text', 'label']]
            dataframes.append(df)
            print(f"         -> {len(df):,} rows added.")
        except Exception as e:
            print(f"  [ERROR] {config['file']}: {e}")

    if not dataframes:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    print("\n[STEP 1] Combining all datasets...")
    df = pd.concat(dataframes, ignore_index=True)
    print(f"         Total rows: {len(df):,}")

    df = df.dropna(subset=['text', 'label'])

    # ── Label mapping ──
    print("[STEP 2] Mapping labels...")
    def map_label(val):
        if isinstance(val, str):
            val = val.lower().strip()
            if 'phish' in val or 'spam' in val or val == '1':
                return 1
            elif 'safe' in val or 'ham' in val or val == '0':
                return 0
        elif isinstance(val, (int, float)):
            return 1 if val > 0 else 0
        return 0

    df['label'] = df['label'].apply(map_label)
    safe_count  = (df['label'] == 0).sum()
    phish_count = (df['label'] == 1).sum()
    print(f"         Safe: {safe_count:,}  |  Phishing: {phish_count:,}")

    # ── Text cleaning ──
    print("[STEP 3] Cleaning text (this may take a few minutes for large datasets)...")
    df['text'] = df['text'].fillna('')
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 5]

    # ── Downsample if too large ──
    MAX_ROWS = 1_000_000
    if len(df) > MAX_ROWS:
        print(f"[STEP 4] Dataset too large ({len(df):,}). Sampling {MAX_ROWS:,} rows...")
        df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"         Final training size: {len(df):,} rows")

    # ── Train/test split ──
    print("[STEP 5] Splitting 80/20...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[['text', 'clean_text']], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # ── TF-IDF vectorization ──
    print("[STEP 6] Building TF-IDF matrix (50,000 features, unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        stop_words='english',
        max_df=0.85,
        min_df=3,
        sublinear_tf=True         # apply log normalization
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw['clean_text'])
    X_test_tfidf  = vectorizer.transform(X_test_raw['clean_text'])
    print(f"         TF-IDF shape: {X_train_tfidf.shape}")

    # ── Handcrafted feature extraction ──
    print("[STEP 7] Extracting structural/behavioral features...")
    X_train_feat = csr_matrix(extract_features(X_train_raw['text']))
    X_test_feat  = csr_matrix(extract_features(X_test_raw['text']))
    print(f"         Structural features per email: {X_train_feat.shape[1]}")

    # ── Combine TF-IDF + structural features ──
    print("[STEP 8] Combining TF-IDF + structural features...")
    X_train_combined = hstack([X_train_tfidf, X_train_feat])
    X_test_combined  = hstack([X_test_tfidf,  X_test_feat])
    print(f"         Combined feature shape: {X_train_combined.shape}")

    # ── Compute class weight for imbalance ──
    ratio = safe_count / max(phish_count, 1)

    # ── Train LightGBM ──
    print("\n[STEP 9] Training LightGBM model...")
    print("         (n_estimators=1000, early stopping on val set)")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=0.1,           # L2 regularization
        scale_pos_weight=ratio,   # handle class imbalance
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    model.fit(
        X_train_combined, y_train,
        eval_set=[(X_test_combined, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True),
                   lgb.log_evaluation(period=100)]
    )

    # ── Evaluate ──
    print("\n[STEP 10] Evaluating model...")
    y_pred = model.predict(X_test_combined)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  [OK] ACCURACY: {accuracy * 100:.2f}%")
    print("  (Model trained with LightGBM + TF-IDF + Structural Features)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)']))

    # ── Save ──
    print("[STEP 11] Saving model artifacts...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir   = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path      = os.path.join(models_dir, 'phish_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

    joblib.dump(model,      model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"  Model saved      -> {model_path}")
    print(f"  Vectorizer saved -> {vectorizer_path}")
    print("\n  [DONE] LightGBM training complete! Model is ready for use.",
          flush=True)
    print("=" * 60)


if __name__ == "__main__":
    train_model()
