
import os
import numpy as np
from sklearn.externals import joblib

# Import all the packages you need for your model below
from flask import Flask, redirect, url_for, render_template, request, flash

clf_model = joblib.load('/models/my_iris_model.pkl')

# Initialise a Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = clf_model.predict(final_features)
    output =prediction[0]
    return render_template('index.html', prediction_text='The Flower is {}'.format(output))


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0')
