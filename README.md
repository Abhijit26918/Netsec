# Network Security — Phishing URL Detection

An end-to-end MLOps pipeline that classifies whether a URL is phishing or legitimate, based on 30 handcrafted URL/page features. Built as a full rebuild of an earlier course project, with a from-scratch pipeline, a live-progress web dashboard, and containerized deployment.

> **Status:** Core ML pipeline, dashboard, and Docker packaging are complete. Cloud sync (AWS S3) and CI/CD (GitHub Actions → ECR → EC2) are in progress — this README will be finalized once that's done.

## Live demo (Docker)

No setup needed — the trained model is baked into the image:

```bash
docker pull chikati/networksecurity:latest
docker run -p 8000:8000 chikati/networksecurity:latest
```

Then open `http://localhost:8000/` for the dashboard, or `http://localhost:8000/docs` for the API reference.

## What it does

```
MongoDB Atlas (raw data)
    │
    ▼
Data Ingestion       — pull from MongoDB, split train/test
    │
    ▼
Data Validation      — schema check + Kolmogorov-Smirnov drift test
    │
    ▼
Data Transformation  — KNN imputation (fit on train only, no leakage)
    │
    ▼
Model Training       — GridSearchCV across 6 classifiers, best picked by accuracy
    │
    ▼
final_model/         — model.pkl + preprocessor.pkl
    │
    ▼
FastAPI + Dashboard  — live training progress, model comparison, predictions
```

## Model performance

Six classifiers are trained and compared via `GridSearchCV` (3-fold CV): Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoost, and XGBoost. The best model by test-set accuracy is what actually serves predictions.

| Model | Test Accuracy |
|---|---|
| **XGBoost** ★ | 97.01% |
| Random Forest | 96.97% |
| Decision Tree | 96.52% |
| Gradient Boosting | 95.57% |
| AdaBoost | 92.36% |
| Logistic Regression | 92.09% |

Final model test set: F1 = 0.973, Precision = 0.964, Recall = 0.983.

XGBoost can be toggled out of the comparison via a single flag (`INCLUDE_XGBOOST` in `networksecurity/components/model_trainer.py`) without touching any other code.

## Dashboard

A custom-built dashboard (not just Swagger UI) at `/`:

- **Live training progress** — click "Run Training Pipeline" and watch all 4 stages update in real time via Server-Sent Events (no polling).
- **Model Selection panel** — every candidate model's accuracy, winner highlighted, with a note confirming exactly which model is being served.
- **Predict panel** — upload a CSV, get results rendered as a table with green "Legitimate" / red "Phishing" pills; large results are paginated client-side rather than dumped as one giant table.

## API

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard UI |
| `GET /docs` | Swagger UI (auto-generated) |
| `GET /train` | Run the full pipeline once, blocking |
| `GET /train-stream` | Run the pipeline with live SSE progress events |
| `GET /metrics` | Latest training run's metrics + model comparison, as JSON |
| `POST /predict` | Upload a CSV (`file` field) → JSON predictions |

## Tech stack

- **ML:** scikit-learn, XGBoost, pandas, numpy
- **Data:** MongoDB Atlas
- **Serving:** FastAPI, Server-Sent Events
- **Packaging:** Docker (non-root user, layered for build-cache efficiency)
- **Planned:** AWS S3 (artifact storage), GitHub Actions → ECR → EC2 (CI/CD)

## Project structure

```
app.py                          FastAPI app + all routes
main.py                         CLI entry point (runs the full pipeline once)
push_data.py                    One-off script: CSV -> MongoDB
Dockerfile
data_schema/schema.yaml         Expected columns for validation
templates/dashboard.html        The dashboard UI

networksecurity/
  exception/                    Custom exception (captures file + line of failure)
  logging/                      One timestamped log file per run
  constant/training_pipeline/   All config constants, single source of truth
  entity/                       Config classes (where to read/write) and
                                 Artifact classes (what each stage produced)
  components/                   data_ingestion, data_validation,
                                 data_transformation, model_trainer
  pipeline/training_pipeline.py Orchestrates all stages
  utils/                        Shared helpers (I/O, metrics, model wrapper)
```

## Running it locally

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# .env: MONGO_DB_URL=<your MongoDB Atlas connection string>

python push_data.py        # one-time: load the dataset into MongoDB
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

## License

Personal / educational project.
