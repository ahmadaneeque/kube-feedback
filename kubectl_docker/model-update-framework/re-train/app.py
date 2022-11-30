
from os import environ, path
from shutil import copyfile
from datetime import datetime

from json import loads, dumps
import pandas as pd

import joblib
from flask import Flask, request, abort

model_path = environ['MODELPATH']

model = None
UPDATE_FLAG = False

app = Flask(__name__)


def load_model(location, name='latest.pkl'):
    """

    :param location:
    :param name:
    :return:
    """
    model = joblib.load(path.join(location, name ))
    model._make_predict_function()
    return model
# @app.route('/hello/<name>')
# @app.route('/login', methods=['POST'])


def make_copy(src):
    modified_time = path.getmtime(src)
    time_in_iso = datetime.utcfromtimestamp(modified_time).\
        strftime('%Y-%m-%d %H:%M:%S')
    dst = path.join(model_path, time_in_iso)
    copyfile(src, dst)


@app.route('/', methods=['POST'])
def home():
    return dumps({'value': 'Hello World this is retraining module'})


@app.route('/anomaly/', methods=['POST'])
def retraining():
    # args = request.args
    args = request.get_json()['timestamps']
    print(args)
    # if 'retraining' == opt :
    # int_features = [float(x) for x in request.form.values()]
    # constant = [x for x in request.form.values()]
    constant = [pd.to_datetime(args['start']),
                pd.to_datetime(args['stop'])]
    file_path = environ['DATAPATH']
    # loads(request.values)

    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=10000, index_col=0)
    df = pd.concat([chunk.loc[constant[0]: constant[1]] for chunk in iter_csv])
    df = df[df.columns.difference(['Grade Code', 'time'])]

    global model
    model = load_model(model_path)
    model.fit(df, df)

    global UPDATE_FLAG
    UPDATE_FLAG = True
    return dumps({'update': 'success'})


@app.route('/save', methods=['POST'])
def update_model():
    if UPDATE_FLAG:
        src = path.join(model_path, 'latest')
        make_copy(src)
        joblib.dump(model, src)

        return dumps({'update': 'success'})
    return dumps({'update': 'no model to save'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
