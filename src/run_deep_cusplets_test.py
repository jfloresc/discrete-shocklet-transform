#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cusplets
import deep_cusplets


def main():
    
    dow30 = pd.read_csv( '../../../../finance-data/dow30_data/data/dow30-19900101-20180202.csv' )

    tickers = set(dow30.ticker)
    ts = [dow30[dow30.ticker == ticker].Volume.values for ticker in tickers]
    ts = np.array( [x for x in ts if len(x) == 7079 ] )  # that's the max length
    feature_dim = 500
    
    Xtrain, ytrain, Xtest, ytest = deep_cusplets.make_indic_regression_data( 
            ts[:-1],
            feature_dim=feature_dim,
            wmax=1500,
            k_args=[3.]
            )
    fit_model = deep_cusplets.create_and_train_mlp_model(Xtrain, ytrain)
    preds = deep_cusplets.predict( fit_model, ts[-1], feature_dim )

    # make the true indicator for comparison
    cc, _ = cusplets.cusplet( 
            ts[-1],
            cusplets.power_cusp,
            np.linspace(10, 1500, 100).astype(int),
            k_args=[3.]
            )
    cusps, cusp_indic, gecusp = cusplets.classify_cusps(cc, b=0.75, geval=0.5)

    plt.plot(cusp_indic[feature_dim:])
    plt.plot(preds)
    plt.show()

if __name__ == "__main__":
    main()
