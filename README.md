# flask-tensorflow-inference-service
flask-tensorflow-inference-service/
# Flask TensorFlow Inference Service

This repo demonstrates a production-style ML inference API using:
- Flask for serving
- TensorFlow model predictions
- Input validation (reject bad requests)
- Request batching (group requests that arrive close together)
- Connection pooling (SQLAlchemy engine for logging predictions)
- Fault-tolerant error handling (logging failures never break serving)

## Why this matters
API responses can look "correct" while the ML layer or downstream logging is broken.
This service shows how to build reliable inference endpoints with testing and safety.

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── batching.py
│   ├── config.py
│   ├── db.py
│   ├── model.py
│   ├── schemas.py
│   └── utils.py
├── scripts/
│   └── create_dummy_model.py
├── tests/
│   ├── test_health.py
│   └── test_predict.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── requirements.txt
├── README.md
├── wsgi.py
└── .env.example
requirements.txt
Flask==3.0.0
tensorflow==2.15.0
numpy==1.26.4
pydantic==2.7.4
SQLAlchemy==2.0.32
psycopg2-binary==2.9.9
pytest==8.2.2
gunicorn==22.0.0

.env.example
FLASK_ENV=production
MODEL_PATH=models/dummy_model.keras

# DB is optional. If not set, DB logging is disabled.
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/inference

# Batching
BATCH_MAX_SIZE=16
BATCH_MAX_WAIT_MS=30
PREDICT_TIMEOUT_MS=2000

wsgi.py
from app import create_app

app = create_app()

app/__init__.py
from flask import Flask, jsonify
from app.api import api_bp
from app.config import Settings
from app.db import init_db
from app.model import load_tf_model
from app.batching import Batcher

def create_app() -> Flask:
    app = Flask(__name__)
    settings = Settings.from_env()

    # Load model (or fail clearly)
    model = load_tf_model(settings.model_path)

    # DB is optional
    engine = init_db(settings.database_url)

    # Start batching worker
    batcher = Batcher(
        model=model,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
        predict_timeout_ms=settings.predict_timeout_ms,
        db_engine=engine,
    )
    batcher.start()

    # Store in app config for routes
    app.config["SETTINGS"] = settings
    app.config["BATCHER"] = batcher

    app.register_blueprint(api_bp)

    # Simple error handling (fault-tolerant)
    @app.errorhandler(Exception)
    def handle_exception(e):
        # Don’t leak stack traces to clients
        return jsonify({
            "ok": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)[:200]
        }), 500

    return app

app/config.py
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    model_path: str
    database_url: str | None
    batch_max_size: int
    batch_max_wait_ms: int
    predict_timeout_ms: int

    @staticmethod
    def from_env() -> "Settings":
        model_path = os.getenv("MODEL_PATH", "models/dummy_model.keras")
        database_url = os.getenv("DATABASE_URL")  # optional

        def to_int(name: str, default: int) -> int:
            v = os.getenv(name)
            return int(v) if v and v.isdigit() else default

        return Settings(
            model_path=model_path,
            database_url=database_url,
            batch_max_size=to_int("BATCH_MAX_SIZE", 16),
            batch_max_wait_ms=to_int("BATCH_MAX_WAIT_MS", 30),
            predict_timeout_ms=to_int("PREDICT_TIMEOUT_MS", 2000),
        )

app/schemas.py (input validation)
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    # Example: model takes a vector of 10 floats
    features: List[float] = Field(..., min_length=10, max_length=10)

class PredictResponse(BaseModel):
    ok: bool
    prediction: float
    model_version: str

app/model.py
import os
import tensorflow as tf

def load_tf_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run: python scripts/create_dummy_model.py"
        )
    model = tf.keras.models.load_model(model_path)
    # Warmup (small) to reduce first-request latency
    _ = model.predict([[0.0]*10], verbose=0)
    return model

app/db.py (connection pooling via SQLAlchemy engine)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

def init_db(database_url: str | None) -> Engine | None:
    if not database_url:
        return None

    # SQLAlchemy provides pooling by default for Postgres.
    # We set a pool size to make it explicit.
    engine = create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

    # Create a simple table if it doesn't exist
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT NOW(),
            features TEXT NOT NULL,
            prediction DOUBLE PRECISION NOT NULL
        )
        """))
    return engine

app/utils.py
import json

def to_json_str(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

app/batching.py (request batching)
import time
import threading
import queue
import numpy as np
from dataclasses import dataclass
from concurrent.futures import Future
from sqlalchemy import text
from sqlalchemy.engine import Engine

@dataclass
class _BatchItem:
    features: list[float]
    future: Future

class Batcher:
    """
    Collects requests and runs model.predict on a batch.
    This improves throughput when many requests arrive close together.
    """
    def __init__(
        self,
        model,
        max_batch_size: int = 16,
        max_wait_ms: int = 30,
        predict_timeout_ms: int = 2000,
        db_engine: Engine | None = None,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.predict_timeout_ms = predict_timeout_ms
        self.db_engine = db_engine

        self._q: "queue.Queue[_BatchItem]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()

    def submit(self, features: list[float]) -> Future:
        fut: Future = Future()
        self._q.put(_BatchItem(features=features, future=fut))
        return fut

    def _run(self):
        while self._running:
            try:
                first = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            start = time.time()

            # Collect until max size or wait time reached
            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - start) * 1000
                remaining = max(0.0, (self.max_wait_ms - elapsed_ms) / 1000)
                if remaining <= 0:
                    break
                try:
                    item = self._q.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            # Run prediction
            try:
                x = np.array([b.features for b in batch], dtype=np.float32)
                y = self.model.predict(x, verbose=0)
                # y shape might be (batch, 1) or (batch,)
                y = np.array(y).reshape(len(batch), -1)

                for i, item in enumerate(batch):
                    pred = float(y[i][0])
                    item.future.set_result(pred)

                # Optional DB logging (fault-tolerant)
                if self.db_engine:
                    try:
                        with self.db_engine.begin() as conn:
                            for item in batch:
                                conn.execute(
                                    text("INSERT INTO inference_logs (features, prediction) VALUES (:f, :p)"),
                                    {"f": str(item.features), "p": float(item.future.result())}
                                )
                    except Exception:
                        # Logging should never break serving
                        pass

            except Exception as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)

app/api.py
from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError
from app.schemas import PredictRequest

api_bp = Blueprint("api", __name__)

@api_bp.get("/health")
def health():
    return jsonify({"ok": True, "status": "up"})

@api_bp.post("/predict")
def predict():
    batcher = current_app.config["BATCHER"]

    # Input validation
    try:
        payload = PredictRequest.model_validate_json(request.data)
    except ValidationError as e:
        return jsonify({
            "ok": False,
            "error": "BAD_INPUT",
            "message": e.errors()[0]["msg"]
        }), 400

    # Submit to batcher and wait result
    fut = batcher.submit(payload.features)

    # Simple timeout handling
    settings = current_app.config["SETTINGS"]
    timeout_s = settings.predict_timeout_ms / 1000.0

    try:
        pred = fut.result(timeout=timeout_s)
    except Exception:
        return jsonify({
            "ok": False,
            "error": "PREDICT_FAILED",
            "message": "Prediction failed or timed out"
        }), 500

    return jsonify({
        "ok": True,
        "prediction": float(pred),
        "model_version": "dummy_v1"
    })

scripts/create_dummy_model.py
import os
import tensorflow as tf

def main():
    os.makedirs("models", exist_ok=True)

    # Simple model: 10 inputs -> 1 output
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # Dummy "training" so it's not random-only
    import numpy as np
    x = np.random.randn(512, 10).astype("float32")
    y = (x.sum(axis=1, keepdims=True) + np.random.randn(512, 1)*0.1).astype("float32")
    model.fit(x, y, epochs=2, verbose=0)

    model_path = "models/dummy_model.keras"
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()

tests/test_health.py
from app import create_app

def test_health():
    app = create_app()
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["ok"] is True

tests/test_predict.py
import os
from app import create_app

def test_predict_valid_input():
    # Make sure model exists
    assert os.path.exists("models/dummy_model.keras"), "Run scripts/create_dummy_model.py first"

    app = create_app()
    client = app.test_client()

    payload = {"features": [0.1]*10}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json["ok"] is True
    assert "prediction" in r.json

def test_predict_bad_input():
    app = create_app()
    client = app.test_client()

    payload = {"features": [0.1]*3}  # wrong length
    r = client.post("/predict", json=payload)
    assert r.status_code == 400

.github/workflows/tests.yml
name: tests

on: [push, pull_request]

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python scripts/create_dummy_model.py
      - run: pytest -q

