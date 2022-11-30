
from os import path
from datetime import datetime
from shutil import copyfile
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
import joblib

app = FastAPI()
model = None


def load_model(location, name='latest.pkl'):
    """
    :param location:
    :param name:
    :return:
    """
    model = joblib.load(path.join(location, name))
    model._make_predict_function()
    return model


def retraining(model_name: str, message=""):
    constant = ['01/08/2019', '30/08/2019']
    # constant = [pd.to_datetime(args['start']),
    #             pd.to_datetime(args['stop'])]
    # file_path = environ['DATAPATH']
    file_path = '../../../../Data_set/packages/anomaly_clean_data_packages.csv'
    # loads(request.values)
    model_path = '../../../../Data_set/packages'
    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=10000, index_col=0)
    df = pd.concat([chunk.loc[constant[0]: constant[1]] for chunk in iter_csv])
    df = df[df.columns.difference(['Grade Code', 'time'])]

    global model
    model = load_model(model_path)
    model.fit(df, df)

    # with open("log.txt", mode="w") as email_file:
    #     content = f"notification for {email}: {message}"
    #     email_file.write(content)


def make_copy(src):
    modified_time = path.getmtime(src)
    time_in_iso = datetime.utcfromtimestamp(modified_time).\
        strftime('%Y-%m-%d %H:%M:%S')
    dst = path.join(src, time_in_iso)
    copyfile(src, dst)


@app.post("/retrain/{model_name}")
async def send_notification(model_name: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(retraining, model_name, message="some notification")
    return {"message": "Notification sent in the background"}


@app.get('/')
def index():
    return {'val': 'hello world'}
