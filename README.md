# Sales Email Analyzer

This project trains two models to classify sales email replies and serves them via a FastAPI endpoint.
- **Baseline**: A simple and fast TF-IDF + Logistic Regression model.
- **Transformer**: A more powerful, fine-tuned `distilbert-base-uncased` model.

## Data
The model expects a CSV file at `data/reply_classification_dataset.csv` with two columns:
- `reply`: The text content of the email reply.
- `label`: The classification (`positive`, `negative`, or `neutral`).

## Setup
1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Train the Models
Before running the API, you must train both models. The artifacts will be saved to the `outputs/` directory.
```bash
# Train the baseline model (fast)
python train_baseline.py

# Train the transformer model (can take a few minutes)
python train_transformer.py
```

### 2. Run the API Server
Once both models are trained, start the FastAPI server.
```bash
python app.py
```
The API will be available at `http://127.0.0.1:8000`.

### 3. Make a Prediction
You can send a POST request to the `/predict` endpoint to get classifications from both models.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Sounds interesting, can you send more details?"}'
```

**Example Response:**
```json
{
  "baseline": {
    "label": "neutral",
    "confidence": 0.6725
  },
  "transformer": {
    "label": "neutral",
    "confidence": 0.4975
  }
}
```
