# Email Reply Classification

This project trains two models on a small email reply dataset:
- Baseline: TF-IDF + Logistic Regression
- Transformer: Fine-tunes `distilbert-base-uncased` with Hugging Face

## Data
Place the CSV at `data/reply_classification_dataset.csv` with columns:
- `reply`: the email reply text
- `label`: one of `positive`, `negative`, `neutral` (case-insensitive)

## Setup
Install Python 3.10+ and dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run (Simple Scripts)
- Baseline:
```bash
python train_baseline.py
```
- Transformer:
```bash
python train_transformer.py
```

Artifacts and metrics are saved under `outputs/` and `models/`.

## API (FastAPI)
Wraps the baseline model for predictions.

1) Ensure the baseline model artifacts exist:
```bash
python train_baseline.py
```

2) Start the API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

3) Predict example:
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
	-H 'Content-Type: application/json' \
	-d '{"text": "Looking forward to the demo!"}'
```
Response:
```json
{ "label": "positive", "confidence": 0.95 }
```