import sys
from flask import Flask, request, render_template
from src.exception import MyException
from src.pipline.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = CustomData(
            gender=request.form.get("gender"),
            married=request.form.get("married"),
            dependents=request.form.get("dependents"),
            education=request.form.get("education"),
            self_employed=request.form.get("self_employed"),
            applicant_income=float(request.form.get("applicant_income")),
            coapplicant_income=float(request.form.get("coapplicant_income")),
            loan_amount=float(request.form.get("loan_amount")),
            loan_amount_term=float(request.form.get("loan_amount_term")),
            credit_history=float(request.form.get("credit_history")),
            property_area=request.form.get("property_area"),
        )

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(data)

        details = {
            "gender": data.gender,
            "married": data.married,
            "dependents": data.dependents,
            "education": data.education,
            "self_employed": data.self_employed,
            "applicant_income": f"{data.applicant_income:,.2f}",
            "coapplicant_income": f"{data.coapplicant_income:,.2f}",
            "loan_amount": f"{data.loan_amount:,.2f}",
            "loan_amount_term": f"{data.loan_amount_term:,.0f}",
            "credit_history": "Yes (1.0)" if data.credit_history == 1.0 else "No (0.0)",
            "property_area": data.property_area,
        }

        return render_template("result.html", prediction=prediction, details=details)

    except Exception as e:
        error_msg = str(MyException(e, sys))
        return render_template("index.html", error=error_msg)


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", error="Page not found.")


@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", error="Internal server error. Please try again later.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
