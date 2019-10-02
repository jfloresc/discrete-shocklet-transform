#!/usr/bin/env python

import os
import pathlib

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

import cusplets
import stat_tools


def make_indic_regression_data( 
        time_series_matrix,
        feature_dim='auto',
        kernel='power_cusp',
        b_mult=0.75,
        geval=0.5,
        wmin=10,
        wmax=500,
        nw=100,
        buff_val='auto',
        k_args=[],
        train_frac=0.8
        ):
    # make everything compatible for what follows
    print('formatting data')
    time_series_matrix = np.array( time_series_matrix )
    if len( time_series_matrix.shape ) == 1:
        time_series_matrix = time_series_matrix.reshape(1, time_series_matrix.shape[0])
    time_series_matrix = time_series_matrix.astype(float)
    np.random.shuffle( time_series_matrix )  # only shuffles rows, leaves time intact

    if feature_dim == 'auto':
        feature_dim = min(500, int(0.25 * time_series_matrix.shape[1]))
    if buff_val == 'auto':
        buff_val = int( 0.25 * wmax )

    indics = []
    x_vars = []
    window_widths = np.linspace( wmin, wmax, nw ).astype(int)
    kernel = getattr(cusplets, kernel)

    print('computing cusplet transforms, this might take awhile')
    for i, x in enumerate( time_series_matrix ):
        cc, _ = cusplets.cusplet(
                              x,  
                              kernel,
                              window_widths,
                              k_args=k_args
                              )
        cusps, cusp_indic, gecusp = cusplets.classify_cusps(cc, b=b_mult, geval=geval)
        indics.append( cusp_indic[buff_val:-buff_val] )  # cut off edge effects 
        x_vars.append( x[buff_val:-buff_val] )  # same here

    indics = np.array( indics )
    x_vars = np.array( x_vars )

    X_vars = np.array( [stat_tools.make_moving_tensor(x_vars[i], feature_dim)\
            for i in range(x_vars.shape[0])] )
    X_vars, mean, std = stat_tools.row_normalize( X_vars )
    X_vars[np.isnan(X_vars)] = 0.
    X_vars = np.vstack( X_vars )

    y_vars = np.array( [indics[j][X_vars.shape[1]:] for j in range(x_vars.shape[0])] )
    y_vars = np.concatenate( y_vars )
    
    train_ind = int( train_frac * len(y_vars) )
    return X_vars[:train_ind], y_vars[:train_ind], X_vars[train_ind:], y_vars[train_ind:]


def classify_cusps(indic, geval=0.5):
    return np.where(indic > geval)


def predict(
        model,
        time_series_matrix,
        feature_dim,
        alg='en'
        ):
    time_series_matrix = np.array( time_series_matrix )
    if len( time_series_matrix.shape ) == 1:
        time_series_matrix = time_series_matrix.reshape(1, time_series_matrix.shape[0])
    time_series_matrix = time_series_matrix.astype(float)
    X_vars = np.array( [stat_tools.make_moving_tensor(time_series_matrix[i], feature_dim)\
            for i in range(time_series_matrix.shape[0])] )
    X_vars, mean, std = stat_tools.row_normalize( X_vars )
    X_vars[np.isnan(X_vars)] = 0.
    X_vars = np.vstack( X_vars )

    return model.predict( X_vars )


def create_and_train_mlp_model(
        X_train,
        y_train,
        verbose=True,
        savemodel=True,
        savedir='./models',
        savename='mlp_model',
        overwrite=True
        ):
    data_dim = X_train.shape[1]
    model = MLPRegressor( 
            hidden_layer_sizes=(int(data_dim / 2), int(data_dim / 4),),
            solver='adam',
            max_iter=500,
            verbose=verbose,
            alpha=0.0025,
            early_stopping=True,
            tol=1e-4,
            learning_rate='adaptive'
            )
    fit_model = model.fit(X_train, y_train)
    if savemodel:
        pathlib.Path( savedir ).mkdir(
            exist_ok=True,
            parents=True
            )
        fname = savedir + '/' + savename + '.joblib.lzma'
        if os.path.exists( fname ) and not overwrite:
            print(f'Not saving model since {fname} exists already')
        else:
            joblib.dump( fit_model, fname, compress=3 )
            print(f'saved model at {fname}')
    return fit_model


def create_and_train_knn_model(
        X_train,
        y_train,
        verbose=True,
        savemodel=True,
        savedir='./models',
        savename='mlp_model',
        overwrite=True
        ):
    data_dim = X_train.shape[1]
    model = KNeighborsRegressor( 
            n_jobs=-1
            )
    fit_model = model.fit(X_train, y_train)
    if savemodel:
        pathlib.Path( savedir ).mkdir(
            exist_ok=True,
            parents=True
            )
        fname = savedir + '/' + savename + '.joblib.lzma'
        if os.path.exists( fname ) and not overwrite:
            print(f'Not saving model since {fname} exists already')
        else:
            joblib.dump( fit_model, fname, compress=3 )
            print(f'saved model at {fname}')
    return fit_model


def create_and_train_adaboost_model(
        X_train,
        y_train,
        verbose=True,
        savemodel=True,
        savedir='./models',
        savename='adaboost_model',
        overwrite=True
        ):
    data_dim = X_train.shape[1]
    model = AdaBoostRegressor( 
            loss='square'
            )
    fit_model = model.fit(X_train, y_train)
    if savemodel:
        pathlib.Path( savedir ).mkdir(
            exist_ok=True,
            parents=True
            )
        fname = savedir + '/' + savename + '.joblib.lzma'
        if os.path.exists( fname ) and not overwrite:
            print(f'Not saving model since {fname} exists already')
        else:
            joblib.dump( fit_model, fname, compress=3 )
            print(f'saved model at {fname}')
    return fit_model


def create_and_train_en_model(
        X_train,
        y_train,
        verbose=True,
        savemodel=True,
        savedir='./models',
        savename='en',
        overwrite=True
        ):
    from sklearn.linear_model import ElasticNetCV
    data_dim = X_train.shape[1]
    model = ElasticNetCV( 
            verbose=True,
            n_jobs=-1,
            selection='random',
            max_iter=3000,
            l1_ratio=np.linspace(0.05, 1, 10),
            cv=3
            )
    fit_model = model.fit(X_train, y_train)
    if savemodel:
        pathlib.Path( savedir ).mkdir(
            exist_ok=True,
            parents=True
            )
        fname = savedir + '/' + savename + '.joblib.lzma'
        if os.path.exists( fname ) and not overwrite:
            print(f'Not saving model since {fname} exists already')
        else:
            joblib.dump( fit_model, fname, compress=3 )
            print(f'saved model at {fname}')
    return fit_model


def create_and_train_sgd_model(
        X_train,
        y_train,
        verbose=True,
        savemodel=True,
        savedir='./models',
        savename='sgd',
        overwrite=True,
        loss='huber'
        ):
    from sklearn.linear_model import SGDRegressor

    data_dim = X_train.shape[1]
    model = SGDRegressor(
            loss=loss,
            verbose=1,
            penalty='elasticnet',
            learning_rate='optimal',
            max_iter=150,
            early_stopping=True,
             )
    print('training svr with sgd')
    fit_model = model.fit(X_train, y_train)
    if savemodel:
        pathlib.Path( savedir ).mkdir(
            exist_ok=True,
            parents=True
            )
        fname = savedir + '/' + savename + '.joblib.lzma'
        if os.path.exists( fname ) and not overwrite:
            print(f'Not saving model since {fname} exists already')
        else:
            joblib.dump( fit_model, fname, compress=3 )
            print(f'saved model at {fname}')
    return fit_model


def create_and_train_model(
        X_train,
        y_train,
        verbose=True,
        savemodel=True,
        savedir='./models',
        savename='mlp_model',
        overwrite=True
        ):
    import keras

    X_train = X_train[..., np.newaxis]

    model = keras.models.Sequential([ 
            keras.layers.Conv1D(32, 3, activation='relu',
                input_shape=(X_train.shape[1], 1)),
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.MaxPooling1D(3),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1, activation='linear'),
            ])

    model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy']
            )
    model.fit(X_train, y_train, epochs=20, batch_size=512, validation_split=0.2)

    if savemodel:
        pathlib.Path( savedir ).mkdir(
            exist_ok=True,
            parents=True
            )
        fname = savedir + '/' + savename + '.joblib.lzma'
        if os.path.exists( fname ) and not overwrite:
            print(f'Not saving model since {fname} exists already')
        else:
            joblib.dump( model, fname, compress=3 )
            print(f'saved model at {fname}')
    return model

