from src.logger import logging
from src.exception import MyException
from src.pipline.training_pipeline import TrainingPipeline
import sys

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        model_trainer_artifact = pipeline.run_pipeline()
        print(f"Model trained with accuracy: {model_trainer_artifact.model_accuracy:.4f}")
        print(f"Model saved at: {model_trainer_artifact.trained_model_file_path}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise MyException(e, sys)
