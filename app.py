import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__, template_folder = 'templates/')

model_path = 'models/model.pkl'

with open (model_path, 'rb') as f:
	model = pickle.load(f)

@app.route("/")
def home():
	return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
	try:
		sepal_length = float(request.form['SepalLength'])
		sepal_width = float(request.form['SepalWidth'])
		petal_length = float(request.form['PetalLength'])
		petal_width = float(request.form['PetalWidth'])

		features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

		prediction = model.predict(features)
        species = ['setosa', 'vensicolor', 'virginica']
        predicted_species  = species[prediction[0]]

		return render_template("index.html", prediction=f'Predicted Iris Species: {predicted_species}')
    
    except Exception as e:
		return render_template("index.html", prediction=f'Error: {str(e)}')


if __name__ == '__main__':
	app.run(debug=True)