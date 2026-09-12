import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        model_trainer_artifact = pipeline.run_pipeline()
        print(model_trainer_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
