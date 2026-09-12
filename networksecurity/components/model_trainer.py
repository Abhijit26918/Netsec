import os
import sys

import mlflow
import mlflow.sklearn
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from networksecurity.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import (
    evaluate_models,
    load_numpy_array_data,
    load_object,
    save_object,
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Flip to False to drop XGBoost from the model comparison without touching the dicts below.
INCLUDE_XGBOOST = True

# Experiment tracking is optional: only activates if these env vars are set (e.g. pointing
# at a Dagshub-hosted MLflow server). No credentials are ever hardcoded here.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(
        self,
        best_model_name: str,
        best_model,
        model_comparison: dict,
        train_metric: ClassificationMetricArtifact,
        test_metric: ClassificationMetricArtifact,
    ):
        """Log this training run to MLflow, if MLFLOW_TRACKING_URI is configured.
        Silently skipped otherwise, so tracking stays optional and the pipeline
        never fails just because experiment tracking isn't set up."""
        if not MLFLOW_TRACKING_URI:
            logging.info("MLFLOW_TRACKING_URI not set — skipping experiment tracking")
            return
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            with mlflow.start_run():
                mlflow.log_param("best_model_name", best_model_name)
                for model_name, accuracy in model_comparison.items():
                    mlflow.log_metric(f"accuracy_{model_name.replace(' ', '_')}", accuracy)

                mlflow.log_metric("train_f1_score", train_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)
                mlflow.log_metric("test_f1_score", test_metric.f1_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                mlflow.sklearn.log_model(best_model, name="model")
            logging.info("Logged this run to MLflow")
        except Exception as e:
            # Tracking is a nice-to-have, not a reason to fail a training run.
            logging.info(f"MLflow tracking failed, continuing without it: {e}")

    def train_model(self, X_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        models = {
            "Random Forest": RandomForestClassifier(verbose=0),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=0),
            "Logistic Regression": LogisticRegression(verbose=0, max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},
            "Random Forest": {"n_estimators": [8, 16, 32, 128, 256]},
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
            "AdaBoost": {"learning_rate": [0.1, 0.01, 0.001], "n_estimators": [8, 16, 32, 64, 128, 256]},
        }

        if INCLUDE_XGBOOST:
            models["XGBoost"] = XGBClassifier(eval_metric="logloss")
            params["XGBoost"] = {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            }

        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models, param=params
        )

        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        logging.info(f"Best model: {best_model_name} with test accuracy: {best_model_score}")

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        self.track_mlflow(
            best_model_name=best_model_name,
            best_model=best_model,
            model_comparison=model_report,
            train_metric=classification_train_metric,
            test_metric=classification_test_metric,
        )

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
            best_model_name=best_model_name,
            model_comparison=model_report,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(x_train, y_train, x_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
