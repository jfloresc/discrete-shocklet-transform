#!/usr/bin/env python

import os

import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import joblib

import cusplets
import deep_cusplets
from shape_detector import word_idx


def train_test_labmt(load=False, alg='elastic_net'):
    np.random.seed( 1 )
    # we were provided with a matlab-style file.....
    mat = io.loadmat('../_data/rank_turbulence_extractorify006_spearman.mat')
    if not os.path.exists( '../_data/labmt_ranks.npz' ):
        print('creating the data')
        ranks = []
        for i, word in enumerate(mat['ok_words']):
            word = word[0][0]
            word_ranks = mat['ok_wordranks'][word_idx(word, mat['ok_words']), :].T
            word_ranks = -np.squeeze(word_ranks)
            ranks.append(word_ranks)

        ranks = np.array(ranks)
        np.savez_compressed( '../_data/labmt_ranks', ranks=ranks )

    else:
        print('loading the data')
        ranks = np.load('../_data/labmt_ranks.npz')['ranks']

    rand_inds = np.random.choice( range(ranks.shape[0]), size=100, replace=False )

    feature_dim = 750
    test_ind = 2000
    wmin = 10
    wmax = 750
    nw = 300

    if load is False:
        Xtrain, ytrain, Xtest, ytest = deep_cusplets.make_indic_regression_data( 
                ranks[rand_inds],
                feature_dim=feature_dim,
                wmin=wmin,
                wmax=wmax,
                nw=nw,
                k_args=[3.],
                train_frac=1.
                )
        
        if alg == 'elastic_net':
            fit_model = deep_cusplets.create_and_train_en_model(Xtrain, ytrain, savename='trainlabmt_testlabmt_en')
        elif alg == 'sgd':
            loss = 'squared_loss'
            fit_model = deep_cusplets.create_and_train_sgd_model(Xtrain, ytrain,
                    savename=f'trainlabmt_testlabmt_sgd{loss}',
                    loss=loss)

    else:
        if alg == 'elastic_net':
            fit_model = joblib.load( './models/trainlabmt_testlabmt_en.joblib.lzma' )
        elif alg == 'sgd':
            loss = 'huber'
            fit_model = joblib.load( f'./models/trainlabmt_testlabmt_sgd{loss}.joblib.lzma' )


    # show results on some random words
    test_inds = np.random.choice( [x for x in np.linspace(0, ranks.shape[0] - 1, ranks.shape[0]).astype(int)\
            if x not in rand_inds],
            size=100, replace=False)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    for i, (ax, ind) in enumerate( zip(axes, test_inds) ):

        testrank = ranks[ind]
        preds = deep_cusplets.predict( fit_model, testrank, feature_dim, alg=alg )

        # make the true indicator for comparison
        cc, _ = cusplets.cusplet( 
                testrank,
                cusplets.power_cusp,
                np.linspace(wmin, wmax, nw).astype(int),
                k_args=[3.]
                )
        cusps, cusp_indic, gecusp = cusplets.classify_cusps(cc, b=0.75, geval=0.5)

        axes[i].plot(cusp_indic[feature_dim:], color='b', linewidth=2)
        axes[i].plot(preds, linewidth=0.5, color='r')
        axes[i].set_title(mat['ok_words'][ind][0][0])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
    
    plt.tight_layout()

    if alg == 'elastic_net':
        plt.savefig('./deep_cusplets_figures/trainlabmt_testlabmt_en.pdf')
        plt.close()

        # now viz the coefs
        coef = fit_model.coef_
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(coef) + 1),
                coef[::-1],
                'k-')
        ax.set_xlabel('$\\tau$ (lags)', fontsize=15)
        ax.set_ylabel('$\\beta_{\\tau}$', fontsize=15)
        ax.tick_params(labelsize=15)
        ax.set_xlim( -10, len(coef) + 1 )

        ax.axhline(0, 0, 1, color='k', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        plt.savefig('./deep_cusplets_figures/trainlabmt_testlabmt_en_coefs.pdf')

    elif alg == 'sgd':
        plt.savefig(f'./deep_cusplets_figures/trainlabmt_testlabmt_sgd{loss}.pdf')
        plt.close()


def plot_traindow_testdow():
    dow30 = pd.read_csv( '../../../../finance-data/dow30_data/data/dow30-19900101-20180202.csv' )

    tickers = sorted(list(set(dow30.ticker)))
    ts = [dow30[dow30.ticker == ticker].Volume.values for ticker in tickers]
    ts = np.array( [x for x in ts if len(x) == 7079 ] )  # that's the max length
    feature_dim = 750
    test_ind = 5000
    wmin = 10
    wmax = 750
    nw = 300

    fit_model = joblib.load( './models/traindow_testdow_mlp_fd700.joblib.lzma' )

    # plot all the test data
    fig_len = np.ceil( np.sqrt(ts.shape[0]) ).astype(int)
    fig, axes = plt.subplots(fig_len, fig_len, figsize=(10, 10) )
    axes = axes.flatten()

    for i in range(len(ts)):

        test_ts = ts[i, test_ind - feature_dim:]  # leave on feature_dim since need it to predict first output
        
        preds = deep_cusplets.predict( fit_model, test_ts, feature_dim )
        # make the true indicator for comparison
        cc, _ = cusplets.cusplet( 
                test_ts,
                cusplets.power_cusp,
                np.linspace(wmin, wmax, nw).astype(int),
                k_args=[3.]
                )
        cusps, cusp_indic, gecusp = cusplets.classify_cusps(cc, b=0.75, geval=0.5)

        axes[i].plot(cusp_indic[feature_dim:], color='b', linewidth=2)
        axes[i].plot(preds, linewidth=1, color='r')

    for i in range(len(ts), fig_len**2):
        axes[i].remove()

    plt.savefig( f'./deep_cusplets_figures/traindow_testdow.pdf' )
    plt.close()

    # now show the actual windows
    fig_len = np.ceil( np.sqrt(ts.shape[0]) ).astype(int)
    fig, axes = plt.subplots(fig_len, fig_len, figsize=(10, 10) )
    axes = axes.flatten()

    for i in range(len(ts)):

        test_ts = ts[i, test_ind - feature_dim:]  # leave on feature_dim since need it to predict first output
        
        preds = deep_cusplets.predict( fit_model, test_ts, feature_dim )
        learned_gecusp = deep_cusplets.classify_cusps( preds, geval=0.2 )
        learned_windows = cusplets.make_components(learned_gecusp, scan_back=100)

        # make the true indicator for comparison
        cc, _ = cusplets.cusplet( 
                test_ts,
                cusplets.power_cusp,
                np.linspace(wmin, wmax, nw).astype(int),
                k_args=[3.]
                )
        cusps, cusp_indic, gecusp = cusplets.classify_cusps(cc, b=0.75, geval=0.5)
        windows = cusplets.make_components(gecusp[0][feature_dim:], scan_back=100)

        for window in windows:
            axes[i].vlines( window, min(test_ts), max(test_ts), color='b',  alpha=0.01)
        for window in learned_windows:
            axes[i].vlines( window, min(test_ts), max(test_ts), color='r',  alpha=0.01)

        axes[i].plot( test_ts[feature_dim:], color='k' )

    for i in range(len(ts), fig_len**2):
        axes[i].remove()

    plt.savefig( f'./deep_cusplets_figures/traindow_testdow_windows.pdf' )
    plt.close()


    
def train_labmt_test_dow30():
    """Train deep cusplet model on dow 30 volume but test on labmt ranks
    """
    # train on labmt word ranks
    pass


if __name__ == "__main__":
    
    train_test_labmt(load=False, alg='elastic_net') 
