
from os import environ, path
from shutil import copyfile
from datetime import datetime

from json import loads, dumps
import pandas as pd
import numpy as np
import joblib

# Import all the packages you need for your model below
from flask import Flask, redirect, url_for, render_template, request, flash

# clf_model = joblib.load('/models/latest.pkl')

model_path = environ['MODELPATH']


def load_model(location, name='latest.pkl'):
    """

    :param location:
    :param name:
    :return:
    """
    model = joblib.load(path.join(location, name ))
    model._make_predict_function()
    return model


clf_model = load_model(model_path)

# Initialise a Flask app
app = Flask(__name__)


@app.route('/', method=['POST'])
def home():
    return dumps({'value': 'Hello World this is scoring module'})


@app.route('/reload/', methods=['POST'])
def reload():
    global clf_model
    clf_model = load_model(model_path)
    return dumps({'action': 'success'})


@app.route('/predict', methods=['POST'])
def predict():
    # int_features = [float(x) for x in request.form.values()]
    int_features = request.get_json()['value']
    final_features = [np.array(int_features)]
    prediction = clf_model.predict(final_features)
    output = prediction[0]
    return dumps({'result': output})
    # return render_template('index.html', prediction_text='The Flower is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
