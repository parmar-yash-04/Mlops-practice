from src.logger import logging
from src.exception import MyException
from src.pipline.training_pipeline import TrainingPipeline
import sys

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        print(f"Model pushed: {artifact.model_pushed}")
        if artifact.model_pushed:
            print(f"Registry path: {artifact.model_registry_path}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise MyException(e, sys)
