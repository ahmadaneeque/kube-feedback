import os
from shutil import copyfile
from datetime import datetime
import time
import pandas as pd
from flask import Flask, jsonify, url_for

from uwsgidecorators import spool
from uwsgi import spooler_jobs
import joblib

from mlog import custom_log

app = Flask(__name__)
# app.secret_key = os.urandom(42)


@app.route("/")
def index():
    return "hello world"
    # return str(url_for('retrain_background'))


@app.route("/retrain/<name>")
def retrain_background(name):
    uid = custom_log.request_uid(name)
    background_task.spool(uid, 'anomaly_clean_data_packages.csv')
    return jsonify({'handle': uid})
    # return "training started", str(dir(val))


@app.route("/list")
def list_spool():
    # all_process = [x.decode("utf-8") .split('/')[-1] for x in spooler_jobs()]
    # print(spooler_jobs())
    # return jsonify(all_process)
    pass
    return 'not implemented'


@app.route("/status/<handle>")
def spool_status(handle):
    history = custom_log.read_logs(handle)
    print(history)
    return str(history[-1]['status'])


@spool(pass_arguments=True)
def background_task(uid, file_name):

    host = '../../../../../Data_set/packages'
    print("Background task triggered with args", file_name)
    # uid = 'a'
    custom_log.write_log(uid, 'in progress', operation='Data loading')
    df = read_data(host, file_name)
    custom_log.write_log(uid, 'in progress', operation='Data loaded')

    custom_log.write_log(uid, 'in progress', operation='Model loading')
    model = load_model(host)
    custom_log.write_log(uid, 'in progress', operation='Model loaded')

    custom_log.write_log(uid, 'in progress', operation='Model training')
    model.fit(df, df)
    custom_log.write_log(uid, 'in progress', operation='Model trained')

    temp_storage = os.path.join(host, uid)
    joblib.dump(model, temp_storage)

    custom_log.write_log(uid, 'completed', operation='Model saved', location=temp_storage)


def load_model(location, name='latest.pkl'):
    """
    :param location:
    :param name:
    :return:
    """
    model = joblib.load(os.path.join(location, name))
    model._make_predict_function()
    return model


def make_copy(src):
    modified_time = os.path.getmtime(src)
    time_in_iso = datetime.utcfromtimestamp(modified_time).\
        strftime('%Y-%m-%d %H:%M:%S')
    dst = os.path.join(src, time_in_iso)
    copyfile(src, dst)


def read_data(host, file_name):

    time_range = ['01/08/2019', '30/08/2019']
    # constant = [pd.to_datetime(args['start']),
    #             pd.to_datetime(args['stop'])]
    # file_path = environ['DATAPATH']
    # print(os.listdir('../../../../../'))
    # file_name =
    file_path = os.path.join(host, file_name)
    # loads(request.values)
    # model_path = '../../../../Data_set/packages'

    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=10000, index_col=0)
    df = pd.concat([chunk.loc[time_range[0]: time_range[1]] for chunk in iter_csv])
    df = df[df.columns.difference(['Grade Code', 'time'])]
    return df
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
