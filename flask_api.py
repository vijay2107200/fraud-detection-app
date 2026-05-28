from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "xgb_model.pkl"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "XGBClassifier"})


@app.get("/features")
def features():
    return jsonify({"features": FEATURES, "count": len(FEATURES)})


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in FEATURES if f not in body]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 422

    try:
        input_df = pd.DataFrame([{f: float(body[f]) for f in FEATURES}])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 422

    prediction = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]

    return jsonify({
        "prediction": "fraud" if prediction == 1 else "legitimate",
        "fraud_probability": round(float(proba[1]), 4),
        "legitimate_probability": round(float(proba[0]), 4),
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
