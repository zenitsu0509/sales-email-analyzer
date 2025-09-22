import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Text Cleaning (copied from train_baseline.py) ---

_MULTISPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']+")

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = _PUNCT.sub(" ", text.strip().lower())
    return _MULTISPACE.sub(" ", t).strip()



app = FastAPI(
    title="Email Reply Classifier API",
    description="Predicts if an email reply is positive, negative, or neutral using two different models.",
    version="1.1.0"
)

# --- Model Loading ---

# Baseline Model
BASELINE_MODEL_DIR = "outputs/baseline"
BASELINE_MODEL_PATH = os.path.join(BASELINE_MODEL_DIR, "model.joblib")
BASELINE_ENCODER_PATH = os.path.join(BASELINE_MODEL_DIR, "label_encoder.joblib")

# Transformer Model
TRANSFORMER_MODEL_DIR = "outputs/transformer"

def load_models():
    """Load all model artifacts or raise an error if they don't exist."""
    # Load Baseline
    if not os.path.exists(BASELINE_MODEL_PATH) or not os.path.exists(BASELINE_ENCODER_PATH):
        raise RuntimeError(f"Baseline model not found in '{BASELINE_MODEL_DIR}'. Run 'python train_baseline.py'.")
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    label_encoder = joblib.load(BASELINE_ENCODER_PATH)

    # Load Transformer
    if not os.path.isdir(TRANSFORMER_MODEL_DIR):
        raise RuntimeError(f"Transformer model not found in '{TRANSFORMER_MODEL_DIR}'. Run 'python train_transformer.py'.")
    transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
    
    return {
        "baseline": (baseline_model, label_encoder),
        "transformer": (transformer_model, tokenizer)
    }

models = load_models()

# --- API Endpoints ---

class PredictRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    label: str
    confidence: float

class PredictResponse(BaseModel):
    baseline: ModelPrediction
    transformer: ModelPrediction

@app.get("/", tags=["General"])
def read_root():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predicts the sentiment of an email reply using both a baseline and a transformer model.
    """
    # Baseline Prediction
    baseline_model, label_encoder = models["baseline"]
    cleaned_text = clean_text(request.text)
    baseline_probs = baseline_model.predict_proba([cleaned_text])[0]
    baseline_pred_idx = baseline_probs.argmax()
    baseline_label = label_encoder.inverse_transform([baseline_pred_idx])[0]
    baseline_confidence = float(baseline_probs[baseline_pred_idx])

    # Transformer Prediction
    transformer_model, tokenizer = models["transformer"]
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = transformer_model(**inputs).logits
    
    transformer_probs = torch.softmax(logits, dim=1)[0]
    transformer_pred_idx = transformer_probs.argmax().item()
    transformer_label = transformer_model.config.id2label[transformer_pred_idx]
    transformer_confidence = float(transformer_probs[transformer_pred_idx])

    return {
        "baseline": {"label": baseline_label, "confidence": baseline_confidence},
        "transformer": {"label": transformer_label, "confidence": transformer_confidence}
    }

# --- Main execution ---

if __name__ == "__main__":
    print("Starting API server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
