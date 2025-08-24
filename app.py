from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("student_performance_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
history_path = "history.csv"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result', methods=["POST"])
def result():
    if request.method == "POST":
        gender = request.form["gender"]
        race = request.form["race"]
        parent_edu = request.form["parent_edu"]
        lunch = request.form["lunch"]
        prep = request.form["prep"]

        # Encode inputs
        gender = encoders['gender'].transform([gender])[0]
        race = encoders['race/ethnicity'].transform([race])[0]
        parent_edu = encoders['parental level of education'].transform([parent_edu])[0]
        lunch = encoders['lunch'].transform([lunch])[0]
        prep = encoders['test preparation course'].transform([prep])[0]

        features = np.array([[gender, race, parent_edu, lunch, prep]])
        prediction = model.predict(features)[0]
        label = encoders['result'].inverse_transform([prediction])[0]

        # Save to history
        new_row = {
            "Gender": request.form["gender"],
            "Race": request.form["race"],
            "Parental_Education": request.form["parent_edu"],
            "Lunch": request.form["lunch"],
            "Prep": request.form["prep"],
            "Prediction": label
        }
        if os.path.exists(history_path):
            pd.concat([pd.read_csv(history_path), pd.DataFrame([new_row])]).to_csv(history_path, index=False)
        else:
            pd.DataFrame([new_row]).to_csv(history_path, index=False)

        return render_template("result.html", prediction=label)

@app.route('/history')
def history():
    if os.path.exists(history_path):
        data = pd.read_csv(history_path).to_dict(orient="records")
    else:
        data = []
    return render_template("history.html", data=data)

if __name__ == '__main__':
    app.run(debug=True)
