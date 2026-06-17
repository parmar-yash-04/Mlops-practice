import os
import sys
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object, load_schema
from src.entity.estimator import TargetValueMapping


class CustomData:
    def __init__(
        self,
        gender: str,
        married: str,
        dependents: str,
        education: str,
        self_employed: str,
        applicant_income: float,
        coapplicant_income: float,
        loan_amount: float,
        loan_amount_term: float,
        credit_history: float,
        property_area: str,
    ):
        self.gender = gender
        self.married = married
        self.dependents = dependents
        self.education = education
        self.self_employed = self_employed
        self.applicant_income = applicant_income
        self.coapplicant_income = coapplicant_income
        self.loan_amount = loan_amount
        self.loan_amount_term = loan_amount_term
        self.credit_history = credit_history
        self.property_area = property_area

    def get_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "Gender": [self.gender],
                "Married": [self.married],
                "Dependents": [self.dependents],
                "Education": [self.education],
                "Self_Employed": [self.self_employed],
                "ApplicantIncome": [self.applicant_income],
                "CoapplicantIncome": [self.coapplicant_income],
                "LoanAmount": [self.loan_amount],
                "Loan_Amount_Term": [self.loan_amount_term],
                "Credit_History": [self.credit_history],
                "Property_Area": [self.property_area],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise MyException(e, sys)


class PredictionPipeline:
    def __init__(self):
        self.schema = load_schema()
        self.target_mapping = TargetValueMapping()

    def _resolve_model_paths(self):
        artifact_dir = os.path.join(os.getcwd(), "artifact")
        model_path = os.path.join(
            artifact_dir, "model_trainer", "trained_model", "model.pkl"
        )
        preprocessor_path = os.path.join(
            artifact_dir, "data_transformation", "transformed_object", "preprocessing.pkl"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run the training pipeline first.")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run the training pipeline first.")
        return model_path, preprocessor_path

    def predict(self, custom_data: CustomData) -> str:
        try:
            df = custom_data.get_dataframe()
            logging.info(f"Input features:\n{df.to_string(index=False)}")

            model_path, preprocessor_path = self._resolve_model_paths()
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            transformed_data = preprocessor.transform(df)
            raw_pred = model.predict(transformed_data)[0]
            result_label = self.target_mapping.inverse_mapping().get(raw_pred, "N")
            logging.info(f"Prediction: raw={raw_pred}, label={result_label}")
            return result_label
        except Exception as e:
            raise MyException(e, sys)
