from src.logger import logging
from src.exception import MyException
from src.pipline.training_pipeline import TrainingPipeline
import sys

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        print(f"Transformed train: {artifact.transformed_train_file_path}")
        print(f"Transformed test: {artifact.transformed_test_file_path}")
        print(f"Preprocessing object: {artifact.preprocessing_obj_file_path}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise MyException(e, sys)
