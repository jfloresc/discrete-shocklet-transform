#!/usr/bin/env python

import numpy as np
import pandas as pd

import deep_cusplets


def main():
    
    dow30 = pd.read_csv( '../../../../finance-data/dow30_data/data/dow30-19900101-20180202.csv' )

    tickers = set(dow30.ticker)
    ts = [dow30[dow30.ticker == ticker].Volume.values for ticker in tickers]
    ts = np.array( [x for x in ts if len(x) == 7079 ] )  # that's the max length
    
    Xtrain, ytrain, Xtest, ytest = deep_cusplets.make_indic_regression_data( 
            ts,
            feature_dim=250,
            wmax=500,
            k_args=[3]
            )
    
    fit_model = deep_cusplets.create_and_train_mlp_model(Xtrain, ytrain)


if __name__ == "__main__":
    main()
