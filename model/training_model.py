from os import environ
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers
from sklearn.externals import joblib


if __name__ == '__main__':

    # file_path = '/home/ahmad/Codes/Data_set/packages/anomaly_clean_data_packages.csv'
    file_path = environ['DATAPATH']
    constant = ast.literal_eval(environ['TIMERANGE'])
        # '2019-07-1', '2019-07-30']
    print(constant)
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

    input_dim = X_train.shape[1]  # the # features
    encoding_dim = 8  # first layer
    hidden_dim = int(encoding_dim / 2)  #hideen layer

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

    checkpointer = ModelCheckpoint(filepath='/logs',
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   mode='max',
                                   save_best_only=True)

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
                              callbacks=[checkpointer, earlystopping]).history

    prediction = autoencoder.predict(X_train)

    joblib.dump(autoencoder, '/model/latest.pkl')
