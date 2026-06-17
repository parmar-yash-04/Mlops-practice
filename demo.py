from src.logger import logging
from src.exception import MyException
from src.pipline.training_pipeline import TrainingPipeline
import sys

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        print(f"Validation report: {artifact.report_file_path}")
        print(f"Validation status: {artifact.validation_status}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise MyException(e, sys)
