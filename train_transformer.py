# train_transformer.py

import os
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split

# --- Text and Label Cleaning (shared with baseline) ---

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

# --- Data Loading ---

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, names=['reply', 'label'], header=0)
    df['reply'] = df['reply'].fillna("")
    df['label'] = df['label'].apply(normalize_label)
    df = df.dropna(subset=['label']).copy()
    # For transformers, we use the raw reply text, not the cleaned one
    return df.reset_index(drop=True)

# --- Main Training Logic ---

def main():
    # Hardcoded settings
    data_path = "data/reply_classification_dataset.csv"
    model_dir = "outputs/transformer"
    model_name = "distilbert-base-uncased"
    epochs = 1
    batch_size = 16
    lr = 5e-5
    test_size = 0.2
    random_state = 42

    # --- Data Prep ---
    print("Loading and preparing data...")
    df = load_dataset(data_path)
    train_df, eval_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

    label_list = ["negative", "neutral", "positive"]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    def to_hf_dataset(df):
        return Dataset.from_dict({
            'text': df['reply'].tolist(),
            'label': [label_to_id[l] for l in df['label'].tolist()],
        })

    train_ds = to_hf_dataset(train_df)
    eval_ds = to_hf_dataset(eval_df)
    
    features = train_ds.features.copy()
    features['label'] = ClassLabel(num_classes=len(label_list), names=label_list)
    train_ds = train_ds.cast(features)
    eval_ds = eval_ds.cast(features)

    # --- Tokenization ---
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Model and Trainer ---
    print("Setting up model and trainer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=dict(enumerate(label_list)), label2id=label_to_id
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
        }

    training_args = TrainingArguments(
        output_dir=model_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=None,
        logging_steps=10,
        seed=random_state,
        fp16=False, # Set to True if you have a CUDA GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Train and Evaluate ---
    print("Starting training...")
    trainer.train()
    
    print("Evaluating final model...")
    eval_metrics = trainer.evaluate()
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=2)
        
    trainer.save_model(model_dir)

    print("\nTransformer Results:")
    print(f"Accuracy: {eval_metrics['eval_accuracy']:.4f}, F1 (Macro): {eval_metrics['eval_f1_macro']:.4f}")

if __name__ == "__main__":
    main()
