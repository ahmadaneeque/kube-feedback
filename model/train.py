<<<<<<< Updated upstream
import numpy as np
import keras
from keras import layers
import numpy as np
from sklearn.externals import joblib
import io
import base64
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# Import all the packages you need for your model below
from flask import Flask, redirect, url_for, render_template, request, flash
from sklearn.externals import joblib
import pandas as pd

# Initialise a Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/hello')
# def index():
#     if 'download' in request.form:
#         return render_template('index.html')
#     # elif 'watch' in request.form:
#     #     return render_template('index.html', prediction_text='The Flower is {}'.format('button 2'))


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig


@app.route('/operation', methods=['POST'])
def train():
    # int_features = [float(x) for x in request.form.values()]
    constant = [x for x in request.form.values()]
    if 'download' in request.form:
        # df = pd.concat([chunk[(chunk['field'] > constant[0]) and (chunk['field'] < constant[0])]
        #                 for chunk in iter_csv])

        fig = Figure()
        ax = fig.subplots()
        ax.plot([1, 2])
        # Save it to a temporary buffer.
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"
    if 'train' in request:
        model = joblib.load('model/lates.pkl')
    # final_features = [np.array(int_features)]
    # prediction = clf_model.predict(final_features)
    # output =prediction[0]
    # # df = pd.concat([chunk[chunk['field'] > constant[0]] for chunk in iter_csv])
        return render_template('index.html', prediction_text='The Flower is {}'.format(str(request.form)))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# def train_autoencoder():
#
#
# if __name__ == "__main__":

    # # This is the size of our encoded representations
    # encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    #
    # # This is our input image
    # input_img = keras.Input(shape=(784,))
    # # "encoded" is the encoded representation of the input
    # encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = layers.Dense(784, activation='sigmoid')(encoded)
    #
    # # This model maps an input to its reconstruction
    # autoencoder = keras.Model(input_img, decoded)

=======
from os import environ
import pandas as pd
import ast

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from sklearn.externals import joblib


if __name__ == '__main__':

    # file_path = '/home/ahmad/Codes/Data_set/packages/anomaly_clean_data_packages.csv'
    file_path = environ['DATAPATH']
    constant = ast.literal_eval(environ['TIMERANGE'])
        # '2019-07-1', '2019-07-30']

    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=1000, index_col=0)
    # k = pd.to_datetime(constant)
    df_final = pd.concat([chunk.loc[constant[0]:constant[1]] for chunk in iter_csv])

    df_final = df_final[df_final.columns.difference(['Grade Code', 'time'])]

    # scaler = MinMaxScaler()
    # values = scaler.fit_transform(df_final)
    # df_final = values
    #Create a test and train sets of our data

    train_perc = int(df_final.shape[0]*.8)
    # # df_final.iloc[:train_perc]

    X_train = df_final.iloc[:train_perc]
    X_test = df_final.iloc[train_perc:]

    input_dim = X_train.shape[1] # the # features
    encoding_dim = 8 # first layer
    hidden_dim = int(encoding_dim / 2) #hideen layer

    nb_epoch = 30
    batch_size = 128
    learning_rate = 0.1

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(encoding_dim, activation='relu')(encoder)
    decoder = Dense(input_dim, activation='tanh')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#    checkpointer = ModelCheckpoint(filepath=file_path,
#                                   save_weights_only=True,
#                                   monitor='val_loss',
#                                   mode='max',
#                                   save_best_only=True)

    earlystopping = EarlyStopping(monitor="val_loss",
                                  min_delta=0,
                                  patience=2,
                                  verbose=0,
                                  mode="auto",
                                  baseline=None,
                                  restore_best_weights=False,)

    history = autoencoder.fit(X_train, X_train,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              verbose=1,
                              callbacks=[earlystopping]).history

    # prediction = autoencoder.predict(X_train)

    joblib.dump(autoencoder, '/model/latest.pkl')
>>>>>>> Stashed changes
