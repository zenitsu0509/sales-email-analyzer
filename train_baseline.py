# train_baseline.py

import json
import os
import re
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Text and Label Cleaning ---

LABEL_NORMALIZATION = {
    'pos': 'positive', 'positive': 'positive',
    'neg': 'negative', 'negative': 'negative',
    'neu': 'neutral',  'neutral': 'neutral',
}

_MULTISPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']+")

def normalize_label(x: str) -> str:
    if not isinstance(x, str): return None
    return LABEL_NORMALIZATION.get(x.strip().lower(), None)

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = _PUNCT.sub(" ", text.strip().lower())
    return _MULTISPACE.sub(" ", t).strip()

# --- Data Loading and Splitting ---

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, names=['reply', 'label'], header=0)
    df['reply'] = df['reply'].fillna("")
    df['label'] = df['label'].apply(normalize_label)
    df = df.dropna(subset=['label']).copy()
    df['reply_clean'] = df['reply'].apply(clean_text)
    return df.reset_index(drop=True)

def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

# --- Model Training and Evaluation ---

def build_baseline_pipeline(max_features: int = 20000, ngram_range=(1, 2)):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("clf", LogisticRegression(max_iter=200))
    ])

def evaluate_classification(y_true: List[str], y_pred: List[str]):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "report": classification_report(y_true, y_pred, digits=3),
    }

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

# --- Main Training Logic ---

def main():
    # Hardcoded settings for simplicity
    data_path = "data/reply_classification_dataset.csv"
    model_dir = "outputs/baseline"
    test_size = 0.2
    random_state = 42

    print("Loading and preprocessing data...")
    df = load_dataset(data_path)
    train_df, test_df = split_data(df, test_size=test_size, random_state=random_state)

    print("Training baseline model...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    
    pipe = build_baseline_pipeline()
    pipe.fit(train_df['reply_clean'], y_train)

    print("Evaluating model...")
    y_pred = pipe.predict(test_df['reply_clean'])
    y_pred_labels = le.inverse_transform(y_pred)
    metrics = evaluate_classification(test_df['label'].tolist(), y_pred_labels)

    print("Saving artifacts...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(model_dir, 'model.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))
    save_json({"metrics": metrics}, os.path.join(model_dir, 'metrics.json'))

    print("\nBaseline Results:")
    print(metrics["report"])
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 (Macro): {metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
