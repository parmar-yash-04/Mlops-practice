import os
import sys
import numpy as np
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.utils.main_utils import load_numpy_array, save_object, read_yaml

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def get_model(self, model_config: dict):
        model_name = model_config.get("model_name", "random_forest")
        random_state = model_config.get("random_state", 101)

        if model_name == "logistic_regression":
            logging.info("Selected model: Logistic Regression")
            return LogisticRegression(
                C=model_config.get("C", 1.0),
                max_iter=model_config.get("max_iter", 1000),
                class_weight="balanced",
                random_state=random_state,
            )
        elif model_name == "decision_tree":
            logging.info("Selected model: Decision Tree")
            return DecisionTreeClassifier(
                max_depth=model_config.get("max_depth", 10),
                min_samples_split=model_config.get("min_samples_split", 7),
                min_samples_leaf=model_config.get("min_samples_leaf", 6),
                criterion=model_config.get("criterion", "entropy"),
                class_weight="balanced",
                random_state=random_state,
            )
        elif model_name == "gradient_boosting":
            logging.info("Selected model: Gradient Boosting")
            return GradientBoostingClassifier(
                n_estimators=model_config.get("n_estimators", 100),
                learning_rate=model_config.get("learning_rate", 0.1),
                max_depth=model_config.get("max_depth", 5),
                min_samples_split=model_config.get("min_samples_split", 7),
                min_samples_leaf=model_config.get("min_samples_leaf", 6),
                random_state=random_state,
            )
        elif model_name == "knn":
            logging.info("Selected model: K-Neighbors")
            return KNeighborsClassifier(
                n_neighbors=model_config.get("n_neighbors", 5),
                weights=model_config.get("weights", "distance"),
            )
        elif model_name == "svm":
            logging.info("Selected model: SVM")
            return SVC(
                C=model_config.get("C", 1.0),
                kernel=model_config.get("kernel", "rbf"),
                class_weight="balanced",
                random_state=random_state,
            )
        else:
            logging.info("Selected model: Random Forest (default)")
            return RandomForestClassifier(
                class_weight="balanced",
                n_estimators=model_config.get("n_estimators", MODEL_TRAINER_N_ESTIMATORS),
                min_samples_split=model_config.get("min_samples_split", MODEL_TRAINER_MIN_SAMPLES_SPLIT),
                min_samples_leaf=model_config.get("min_samples_leaf", MODEL_TRAINER_MIN_SAMPLES_LEAF),
                max_depth=model_config.get("max_depth", None),
                criterion=model_config.get("criterion", "entropy"),
                random_state=random_state,
            )

    def train_model(self, x_train, y_train):
        try:
            model_config = read_yaml(self.model_trainer_config.model_config_file_path)
            logging.info(f"Model config: {model_config}")

            model = self.get_model(model_config)
            model.fit(x_train, y_train)
            logging.info(f"{type(model).__name__} trained successfully")
            return model
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Trainer")

            train_arr = load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Loaded train array: {train_arr.shape}")
            logging.info(f"Loaded test array: {test_arr.shape}")

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info(f"Train features: {x_train.shape}, Train target: {y_train.shape}")
            logging.info(f"Test features: {x_test.shape}, Test target: {y_test.shape}")

            model_config = read_yaml(self.model_trainer_config.model_config_file_path)
            model_name = model_config.get("model_name", "random_forest")

            mlflow.log_params({
                "model_type": model_name,
                "random_state": model_config.get("random_state", 101),
                "smote_applied": True,
            })
            model_specific_params = {k: v for k, v in model_config.items() if k not in ["model_name", "random_state"]}
            mlflow.log_params(model_specific_params)

            smote = SMOTE(random_state=model_config.get("random_state", 101))
            x_train, y_train = smote.fit_resample(x_train, y_train)
            unique, counts = np.unique(y_train, return_counts=True)
            logging.info(f"After SMOTE - classes: {dict(zip(unique, counts))}")
            mlflow.log_metrics({
                "smote_samples_majority": int(counts.max()),
                "smote_samples_minority": int(counts.min()),
            })

            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_train_prob = model.predict_proba(x_train)[:, 1]
            train_accuracy = accuracy_score(y_train, y_train_pred)
            logging.info(f"Train accuracy: {train_accuracy:.4f}")

            y_test_prob = model.predict_proba(x_test)[:, 1]

            precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_prob)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            best_threshold = thresholds[np.argmax(f1_scores)]
            logging.info(f"Optimal decision threshold: {best_threshold:.4f}")

            y_test_pred = (y_test_prob >= best_threshold).astype(int)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_prob)
            logging.info(f"Test accuracy: {test_accuracy:.4f}")
            logging.info(f"Test precision: {test_precision:.4f}")
            logging.info(f"Test recall: {test_recall:.4f}")
            logging.info(f"Test F1-score: {test_f1:.4f}")
            logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")

            mlflow.log_metrics({
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "test_precision": round(test_precision, 4),
                "test_recall": round(test_recall, 4),
                "test_f1": round(test_f1, 4),
                "test_roc_auc": round(test_roc_auc, 4),
                "best_threshold": round(best_threshold, 4),
            })

            cm = confusion_matrix(y_test, y_test_pred)
            logging.info(f"Confusion matrix:\n{cm}")
            logging.info(f"Classification report:\n{classification_report(y_test, y_test_pred, target_names=['N', 'Y'])}")

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["N", "Y"], yticklabels=["N", "Y"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            report_str = classification_report(y_test, y_test_pred, target_names=["N", "Y"])
            report_path = os.path.join(self.model_trainer_config.trained_model_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report_str)
            mlflow.log_artifact(report_path, artifact_path="metrics")

            model_accuracy = test_accuracy

            if model_accuracy < self.model_trainer_config.expected_score:
                logging.warning(
                    f"Model accuracy {model_accuracy:.4f} is below expected threshold "
                    f"{self.model_trainer_config.expected_score}"
                )

            trained_model_dir = self.model_trainer_config.trained_model_dir
            trained_model_path = self.model_trainer_config.trained_model_path
            os.makedirs(trained_model_dir, exist_ok=True)
            save_object(trained_model_path, model)
            logging.info(f"Model saved: {trained_model_path}")

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=x_test[:5],
            )
            mlflow.log_artifact(trained_model_path, artifact_path="model")
            logging.info("Model Trainer completed successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=trained_model_path,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                model_accuracy=model_accuracy,
            )

        except Exception as e:
            raise MyException(e, sys)