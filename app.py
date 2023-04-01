import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='template')


@app.route("/")
def loadPage():
    return render_template('home.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
    # Retrieve input values
    input_values = [request.form[f"query{i}"] for i in range(1, 6)]

    # Load the trained model
    model = pickle.load(open("PumpDiagnosticModel.pkl", "rb"))

    # Create a DataFrame from user input
    data = {"mean": input_values[0],
            "max": input_values[1],
            "kurtosis": input_values[2],
            "variance": input_values[3],
            "onenorm": input_values[4]}
    new_df = pd.DataFrame(data, index=[0])

    # Make a prediction and calculate the probability
    prediction = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1]

    # Determine the pump condition and confidence
    pump_conditions = ["health", "severe", "mild", "unstable"]
    condition = pump_conditions[prediction[0]]
    confidence = probability[0] * 100

    # Render the results on the home page
    output1 = f"The pump is in {condition} condition"
    output2 = f"Confidence: {confidence}"

    return render_template('home.html', output1=output1, output2=output2, **request.form)


if __name__ == "__main__":
    app.run()
