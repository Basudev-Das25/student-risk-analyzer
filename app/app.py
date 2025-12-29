from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(
    __name__,
    template_folder="template",
    static_folder="static"
)

#Absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

#load model and encoder
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        print("POST RECEIVED")

        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        previous_score = float(request.form["previous_score"])
        assignments_completed = float(request.form["assignments_completed"])
        internal_marks = float(request.form["internal_marks"])

        print("INPUT VALUES:", hours, attendance, previous_score, assignments_completed, internal_marks)

        input_data = pd.DataFrame(
            [[hours, attendance, previous_score, assignments_completed, internal_marks]],
                columns=[
                    "hours_studied",
                    "attendance",
                    "previous_score",
                    "assignments_completed",
                    "internal_marks",
                ],
        )

        encoded_prediction = model.predict(input_data)[0]
        print("ENCODED PREDICTION:", encoded_prediction)

        pridiction = label_encoder.inverse_transform([encoded_prediction])[0]
        print("FINAL PREDICTION:", pridiction)
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run()