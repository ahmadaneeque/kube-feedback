<<<<<<< Updated upstream

import io
import os
import base64
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# Import all the packages you need for your model below
from flask import Flask, redirect, url_for, render_template, request, flash
import pandas as pd

import numpy as np
import keras
from keras import layers
import numpy as np
from sklearn.externals import joblib

model = joblib.load('model/latest.pkl')

model._make_predict_function()

app = Flask(__name__)

=======
import io
import base64
import random
from os import environ, path

from json import loads, dumps
import pandas as pd
from sklearn.externals import joblib

from flask import Response
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
# Import all the packages you need for your model below

from flask import Flask, redirect, url_for, render_template, request, flash

model_path = environ['MODELPATH']

model = joblib.load(path.join(model_path, 'latest.pkl'))
model._make_predict_function()

UPDATE_FLAG = False
app = Flask(__name__)


>>>>>>> Stashed changes
@app.route('/')
def home():
    return render_template('index.html')


<<<<<<< Updated upstream
=======
# @app.route('/plot.png')
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig


>>>>>>> Stashed changes
@app.route('/operation', methods=['POST'])
def operation():
    # int_features = [float(x) for x in request.form.values()]
    constant = [x for x in request.form.values()]
    file_path = '../../Data_set/packages/anomaly_clean_data_packages.csv'
    # df = pd.read_csv(file_path, parse_dates=[0], low_memory=False)
<<<<<<< Updated upstream
    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=100000, index_col=0)
    df = pd.concat([chunk.loc[constant[0]: constant[1]] for chunk in iter_csv])
    df = df[df.columns.difference(['Grade Code', 'time'])]

    if 'plot' in request.form:
        fig = Figure(figsize=[20, 30])
        ax = fig.subplots()
        df.plot(ax=ax, subplots=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        fig.close()
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        del df
        return render_template('index.html', figure_to_plot=f'data:image/png;base64,{data}',
                               prediction_text=constant)

    elif 'score' in request.form:
        result = pd.DataFrame(model.predict(df), columns=df.columns, index=df.index)

        fig, axs = plt.subplots(nrows=len(df.columns), ncols=1, sharex=True, figsize=[20, 30])
        for index, name in enumerate(df.columns):
            axs[index].plot(df[name], label=name)
            axs[index].plot(result[name], label=f'predicted {name}')
            axs[index].legend(loc='upper right')

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        fig.close()
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        del df
        return render_template('index.html', figure_to_plot=f'data:image/png;base64, {data}',
                               prediction_text=constant)

    elif 'train' in request.form:
        model.fit(df, df)
        del df
        return render_template('index.html')
        #                        figure_to_plot='<img src="/plot" alt="my plot">' )
       # return render_template('index.html', prediction_text='The Flower is {}'.format(str(request.form)))
    # return render_template('index.html', prediction_text='The Flower is {}'.format(str(request.form)))
=======
    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=10000, index_col=0)
    df = pd.concat([chunk.loc[constant[0]: constant[1]] for chunk in iter_csv])
    df = df[df.columns.difference(['Grade Code', 'time'])]

    # if 'plot' in request.form:
    #     fig = Figure(figsize=[20, 30])
    #     ax = fig.subplots()
    #     df.plot(ax=ax, subplots=True)
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png")
    #     fig.close()
    #     # Embed the result in the html output.
    #     data = base64.b64encode(buf.getbuffer()).decode("ascii")
    #     return render_template('index.html', figure_to_plot=f'data:image/png;base64,{data}',
    #                            prediction_text=constant)

    # elif 'score' in request.form:
    #     result = pd.DataFrame(model.predict(df), columns=df.columns, index=df.index)
    #
    #     fig, axs = plt.subplots(nrows=len(df.columns), ncols=1, sharex=True, figsize=[20, 30])
    #     for index, name in enumerate(df.columns):
    #         axs[index].plot(df[name], label=name)
    #         axs[index].plot(result[name], label=f'predicted {name}')
    #         axs[index].legend(loc='upper right')
    #
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png")
    #     fig.close()
    #     # Embed the result in the html output.
    #     data = base64.b64encode(buf.getbuffer()).decode("ascii")
    #     return render_template('index.html', figure_to_plot=f'data:image/png;base64, {data}',
    #                            prediction_text=constant)

    # elif 'train' in request.form:
    if 'train' in request.form:

        model.fit(df, df)
        global UPDATE_FLAG
        UPDATE_FLAG = True
        return dumps({'success'})
        # return render_template('index.html')
        #                        figure_to_plot='<img src="/plot" alt="my plot">' )
# return render_template('index.html', prediction_text='The Flower is {}'.format(str(request.form)))


@app.route('/save', methods=['POST'])
def update_model():
    if UPDATE_FLAG:
        return render_template('index.html', prediction_text='Model updater')

    return render_template('index.html', prediction_text='No Model to Update')
>>>>>>> Stashed changes


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
