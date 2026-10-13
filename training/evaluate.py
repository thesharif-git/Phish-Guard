import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import clean_text, extract_features
from scipy.sparse import hstack, csr_matrix

def evaluate_model():
    print("="*50)
    print("AI Phishing Detector - Model Evaluation")
    print("="*50)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    model_path = os.path.join(models_dir, 'phish_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Trained model or vectorizer not found.")
        print(f"Please run 'python training/train_model.py' first.")
        sys.exit(1)
        
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Models loaded successfully!")
    
    csv_path = r"C:\Users\shari\OneDrive\Desktop\Phishing_Email.csv"
    
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found. Running benchmark on sample test cases...")
        sample_texts = [
            "URGENT: Your PayPal account has been suspended! Click http://paypa1-verify.xyz immediately.",
            "Hey team, here is the updated meeting agenda for tomorrow's sync. Best regards.",
            "Your Microsoft 365 password expires today! Confirm your billing at http://microsoft-365verify.net",
            "Attached is the project status report for Q3 review. Thanks!",
            "UNAUTHORIZED ACCESS DETECTED! Act now to avoid account termination: http://bit.ly/secure-login"
        ]
        sample_labels = [1, 0, 1, 0, 1]
        df = pd.DataFrame({'text': sample_texts, 'label': sample_labels})
    else:
        print("\nLoading dataset for evaluation...")
        df = pd.read_csv(csv_path).sample(n=min(20000, len(pd.read_csv(csv_path))), random_state=123)
        if 'Email Text' in df.columns and 'Email Type' in df.columns:
            df = df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})
        else:
            df.columns = ['text', 'label'] + list(df.columns[2:])
            
        df = df.dropna(subset=['text', 'label'])
        
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
    
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 5]
    
    y_test = df['label']
    
    print("\nVectorizing text & extracting structural features...")
    X_test_tfidf = vectorizer.transform(df['clean_text'])
    X_test_feat = csr_matrix(extract_features(df['text']))
    X_test_combined = hstack([X_test_tfidf, X_test_feat])
    
    print("Predicting with combined feature vectors...")
    y_pred = model.predict(X_test_combined)
    y_prob = model.predict_proba(X_test_combined)[:, 1] # Probability of being phishing
    
    print("\n--- Evaluation Results ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    pd_cm = pd.DataFrame(cm, index=['Actual Safe', 'Actual Phishing'], columns=['Predicted Safe', 'Predicted Phishing'])
    print(pd_cm)
    
    print("\nSample Probabilities (Top 5 most confident Phishing predictions):")
    phishing_indices = y_prob.argsort()[::-1][:5]
    for idx in phishing_indices:
        real_label = "Phishing" if y_test.iloc[idx] == 1 else "Safe"
        text_snippet = df['text'].iloc[idx][:100].replace('\n', ' ') + "..."
        print(f"Prob: {y_prob[idx]:.4f} | Actual: {real_label} | Text: {text_snippet}")

if __name__ == "__main__":
    evaluate_model()
