"""REST API that serves the SMS spam model.

Run:   python api.py                 (development server on http://127.0.0.1:5000)
Call:  curl -X POST http://127.0.0.1:5000/classify -H "Content-Type: application/json" \
            -d '{"message": "WINNER!! Call 09061701461 now to claim your free prize"}'
"""
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"
MAX_CHARS = 2000


def create_app(model=None):
    app = Flask(__name__)
    app.config["MODEL"] = model if model is not None else joblib.load(MODEL_PATH)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/classify")
    def classify():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "message" not in payload:
            return jsonify(error="Send JSON like {\"message\": \"...\"}"), 400
        text = payload["message"]
        if not isinstance(text, str) or not text.strip():
            return jsonify(error="'message' must be a non-empty string"), 400
        if len(text) > MAX_CHARS:
            return jsonify(error=f"'message' is longer than {MAX_CHARS} characters"), 413
        prob = float(app.config["MODEL"].predict_proba([text])[0, 1])
        return jsonify(spam_probability=round(prob, 4), label="spam" if prob >= 0.5 else "ham")

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))
