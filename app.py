import asyncio
import json
import os
import queue
import sys
import threading
from dataclasses import asdict

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

load_dotenv()

METRICS_FILE = "final_model/latest_metrics.json"
PREDICT_PREVIEW_LIMIT = 200

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/train")
async def train_route():
    """Blocking, one-shot training run (used by CI/CD, no live progress)."""
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return {"message": "Training is successful"}
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.get("/train-stream")
async def train_stream():
    """
    Server-Sent Events endpoint: runs the pipeline in a background thread
    (training is CPU-bound and would block the async event loop otherwise)
    and streams a JSON event after every stage transition.
    """
    event_queue: "queue.Queue" = queue.Queue()

    def on_event(event: dict):
        event_queue.put(event)

    def worker():
        try:
            pipeline = TrainingPipeline()
            artifact = pipeline.run_pipeline_with_progress(on_event=on_event)
            metrics = {
                "timestamp": pipeline.training_pipeline_config.timestamp,
                "train_metric": asdict(artifact.train_metric_artifact),
                "test_metric": asdict(artifact.test_metric_artifact),
                "best_model_name": artifact.best_model_name,
                "model_comparison": artifact.model_comparison,
            }
            os.makedirs("final_model", exist_ok=True)
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
        finally:
            event_queue.put(None)  # sentinel: stream is done

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, event_queue.get)
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/metrics")
async def get_metrics():
    if not os.path.exists(METRICS_FILE):
        return JSONResponse({"available": False})
    with open(METRICS_FILE) as f:
        data = json.load(f)
    data["available"] = True
    return JSONResponse(data)


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        total_rows = len(df)
        phishing_count = int((df["predicted_column"] == 1).sum())
        legit_count = int((df["predicted_column"] == 0).sum())

        # Cap the JSON payload so a large upload can't freeze the browser rendering it,
        # while the full result set is still on disk at prediction_output/output.csv.
        preview_df = df.head(PREDICT_PREVIEW_LIMIT)
        preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

        return JSONResponse(
            {
                "columns": preview_df.columns.tolist(),
                "rows": preview_df.values.tolist(),
                "total_rows": total_rows,
                "truncated": total_rows > len(preview_df),
                "phishing_count": phishing_count,
                "legit_count": legit_count,
                "output_path": "prediction_output/output.csv",
            }
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
