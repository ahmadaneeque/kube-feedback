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

