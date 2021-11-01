import pickle
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import pandas as pd

app = Flask(__name__)


@cross_origin()
@app.route('/', methods=['GET'])
def homepage():
    return render_template('page1.html')


@cross_origin()
@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@cross_origin()
@app.route('/AppInfo', methods=['GET'])
def appinfo():
    return render_template('AppInfo.html')


@cross_origin()
@app.route('/contact', methods=['GET'])
def developer():
    return render_template('Developer.html')


@cross_origin()
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    with open("scaler_model.pkl", 'rb') as f:
        scalar = pickle.load(f)

    with open("Final_ModelForPrediction.pkl", 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        fixed_acidity = float(request.form['Fixed Acidity'])
        volatile_acidity = float(request.form["Volatile Acidity"])
        citric_acid = float(request.form['Citric Acid'])
        residual_sugar = float(request.form["Residual Sugar"])
        chlorides = float(request.form['Chlorides'])
        free_sulfur_dioxide = float(request.form["Free Sulfur Dioxide"])
        total_sulfur_dioxide = float(request.form['Total Sulfur Dioxide'])
        density = float(request.form["Density"])
        pH = float(request.form['pH'])
        sulphate = float(request.form["Sulphates"])
        alcohol = float(request.form['Alcohol'])
        d = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
              total_sulfur_dioxide, density, pH, sulphate, alcohol]]
        scaled_data = scalar.transform(d)
        prediction = model.predict(scaled_data)
        predict = prediction
        print(prediction)

        if predict[0] <= 5:
            return render_template('output.html', result="Wine Quality Is Average")
        elif 5 < predict[0] <= 7:
            return render_template('output.html', result="Wine Quality Is Good")
        else:
            return render_template('output.html', result="Wine Quality Is Awesome")
    else:
        return render_template('page1.html')


if __name__ == "__main__":
    app.run(debug=True)
    host = '0.0.0.0'
    port = 5000
