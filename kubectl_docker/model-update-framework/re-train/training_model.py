
from os import environ
import pandas as pd
import ast
import random  

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from sklearn.externals import joblib


if __name__ == '__main__':

    random.seed(0)
    # file_path = '/home/ahmad/Codes/Data_set/packages/anomaly_clean_data_packages.csv'
    file_path = environ['DATAPATH']
    constant = ast.literal_eval(environ['TIMERANGE'])

    iter_csv = pd.read_csv(file_path, parse_dates=[0], iterator=True, chunksize=1000, index_col=0)
    # k = pd.to_datetime(constant)
    df_final = pd.concat([chunk.loc[constant[0]:constant[1]] for chunk in iter_csv])

    df_final = df_final[df_final.columns.difference(['Grade Code', 'time'])]

    train_perc = int(df_final.shape[0]*.8)   # because it is a time series
    # # df_final.iloc[:train_perc]

    X_train = df_final.iloc[:train_perc]
    X_test = df_final.iloc[train_perc:]

    autoencoder = joblib.load('/model/latest.pkl')

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
    y_pred = autoencoder.predict(X_test)


    print('mse-val - ',mean_squared_error(X_test.values, y_pred))


