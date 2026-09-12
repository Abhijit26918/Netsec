import sys
from dataclasses import asdict

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config
            )
            logging.info("Starting data validation")
            artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config
            )
            logging.info("Starting data transformation")
            artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact
            )
            logging.info("Starting model training")
            artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline_with_progress(self, on_event=None) -> ModelTrainerArtifact:
        """
        Same four stages as run_pipeline(), but calls on_event({"stage", "status", "detail"})
        before/after each stage so a caller (e.g. an SSE endpoint) can report live progress.
        Kept separate from run_pipeline() so plain callers (main.py, GET /train) are unaffected.
        """

        def emit(stage: str, status: str, detail: dict = None):
            if on_event:
                on_event({"stage": stage, "status": status, "detail": detail or {}})

        current_stage = "pipeline"
        try:
            current_stage = "data_ingestion"
            emit(current_stage, "running")
            data_ingestion_artifact = self.start_data_ingestion()
            train_rows = sum(1 for _ in open(data_ingestion_artifact.trained_file_path, encoding="utf-8")) - 1
            test_rows = sum(1 for _ in open(data_ingestion_artifact.test_file_path, encoding="utf-8")) - 1
            emit(current_stage, "done", {"train_rows": train_rows, "test_rows": test_rows})

            current_stage = "data_validation"
            emit(current_stage, "running")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            emit(current_stage, "done", {"validation_status": data_validation_artifact.validation_status})

            current_stage = "data_transformation"
            emit(current_stage, "running")
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            emit(current_stage, "done", {})

            current_stage = "model_trainer"
            emit(current_stage, "running")
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            emit(
                current_stage,
                "done",
                {
                    "train_metrics": asdict(model_trainer_artifact.train_metric_artifact),
                    "test_metrics": asdict(model_trainer_artifact.test_metric_artifact),
                    "best_model_name": model_trainer_artifact.best_model_name,
                    "model_comparison": model_trainer_artifact.model_comparison,
                },
            )

            emit("pipeline", "complete", {"artifact_dir": self.training_pipeline_config.artifact_dir})
            return model_trainer_artifact
        except Exception as e:
            emit(current_stage, "failed", {"error": str(e)})
            emit("pipeline", "failed", {"error": str(e)})
            raise NetworkSecurityException(e, sys)
