#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
from scipy import io, stats, signal


def word_idx(word, arr):
    return np.where(arr == word)[0]


def _prob_featues_nonparametric_multi(arr,
                  w_size,
                  shock_w_size,
                  min_std=1,
                  max_std=3,
                  diff_thresh=1,
                  lower_thresh=5,
                  shock_thresh=1,
                  shock_lower_thresh=1000):

    # find std of each
    weights = [np.std(a for a in arr)]
    weights = weights / np.sum(weights)

    res = [prob_features_nonparametric(a,
                  w_size,
                  shock_w_size,
                  min_std=1,
                  max_std=3,
                  diff_thresh=1,
                  lower_thresh=5,
                  shock_thresh=1,
                  shock_lower_thresh=1000,
                  multi=False)\
            for a in arr]
    return res, weights
    

def prob_features_nonparametric(arr,
                  w_size,
                  shock_w_size,
                  min_std=1,
                  max_std=3,
                  diff_thresh=1,
                  lower_thresh=5,
                  shock_thresh=1,
                  shock_lower_thresh=1000,
                  multi=False):
    """
    .. function:: prob_features_nonparametric(arr,
        w_size,
        shock_w_size[,
        min_std=0.2,
        max_std=2,
        diff_thresh=3,
        lower_thresh=3,
        shock_thresh=5,
        shock_lower_thresh=5,
        multi=False])
        Filter time series with a nonparametric digital filter to detect cusps and shocks.

        :param arr: numpy array (time series) shape (n,) or (n,1)
        :param w_size: window length corresponding to characteristic timescale of cusps
        :param shock_w_size: window legnth corresponding to characteristic timescale of shocks
        :param min_std: minimum multiple of standard deviation used for skeletonization
        :param max_std: maximum multiple of standard deviation used for skeletonization
        :param diff

        This is a nonlinear digital filter to extract shocks and cusps from time series data. 
        It is timescale-independent and does not rely on any machine learning techniques. 
        It is not designed for dynamic (online / real-time) use but in practice has 
        proved useful in this task.
    """

    if multi is True:
        assert len(arr.shape) == 2
        return _prob_features_nonparametric_multi(arr,
                  w_size,
                  shock_w_size,
                  min_std=min_std,
                  max_std=max_std,
                  diff_thresh=diff_thresh,
                  lower_thresh=lower_thresh,
                  shock_thresh=shock_thresh,
                  shock_lower_thresh=shock_lower_thresh)
    
    w_size += w_size % 2 + 1
    
    stds_mults = np.linspace(min_std, max_std, int(np.sqrt(len(arr))))
        
    ksize = int(np.sqrt(len(arr)))
    ksize += ksize % 2 - 1  # ensure kernel is odd and err on small side
    med_arr = signal.medfilt(arr,
                            kernel_size=ksize)
    diff_med = np.diff(med_arr)
    diff_med_std = np.std(diff_med)
    signal_nums = np.zeros( (stds_mults.shape[0], len(diff_med) ) )
    
    for i, s in enumerate(stds_mults):
        signal_nums[i] = np.where(np.absolute(diff_med) >= s * diff_med_std * np.ones(len(diff_med)),
                                 np.sign(diff_med),
                                 0)
    agg_sn = np.sum(signal_nums, axis=0)
    cusps = []
    shocks = []
    
    # now classify
    for i in range(len(agg_sn) - w_size):
        window = agg_sn[i : i + w_size - 1]
        pt_ind = i + w_size // 2
        
        if i > w_size // 2:
            rev_pt_ind = i - w_size // 2
        else:
            rev_pt_ind = i
            
        pt = window[len(window) // 2]
        
        sum_first_half = np.sum(window[ : len(window)//2 - 1])
        sum_second_half = np.sum(window[len(window)//2 + 1: ])
        
        if (np.sum(window) < diff_thresh)\
            and (np.abs(sum_first_half) >= lower_thresh)\
            and (np.abs(sum_second_half) >= lower_thresh):
                cusps.extend(list(range(rev_pt_ind, i + w_size - 1)))
                
    for i in range(len(agg_sn) - shock_w_size):
        window = agg_sn[i : i + shock_w_size - 1]
        arr_window = arr[i : i + shock_w_size -1]
        pt_ind = i + shock_w_size // 2
        
        if i > shock_w_size // 2:
            rev_pt_ind = i - shock_w_size // 2
        else:
            rev_pt_ind = i
            
        pt = window[len(window) // 2]
        
        sum_first_half = np.sum(window[ : len(window)//2 - 1])
        sum_second_half = np.sum(window[len(window)//2 + 1 : ])
        mean_first_half_arr = np.mean(arr_window[ : len(arr_window)//2 - 1])
        mean_second_half_arr = np.mean(arr_window[len(arr_window)//2 + 1 : ])
                
        if (np.abs(sum_first_half) - np.abs(sum_second_half) < shock_thresh)\
            and (np.absolute(mean_first_half_arr - mean_second_half_arr) > shock_lower_thresh):
                shocks.extend(list(range(rev_pt_ind, i + shock_w_size - 1)))
                
    cusps = np.array(cusps)
    shocks = np.array(shocks)
            
    return (med_arr, signal_nums, cusps, shocks,
            np.array(list(range(len(agg_sn) - w_size))),
            np.array(list(range(len(agg_sn) - shock_w_size))))


if __name__ == "__main__":

    mat = io.loadmat('rank_turbulence_extractorify006_spearman.mat')
    n_words = len(mat['ok_words'].flatten())
    
    for i, word in enumerate(mat['ok_words'].flatten()):
        if i % 100 == 0:
            print('on word {} of {}'.format(i, n_words))
        arr = mat['ok_wordranks'][word_idx(word[0], mat['ok_words']), :].T
        arr = np.squeeze(arr)

        med, agg, cusps, shocks, cusp_range, shock_range = prob_features_nonparametric(-arr,
                        90,
                        90,
                        min_std=1,
                        max_std=3,
                        diff_thresh=1,
                        lower_thresh=5,
                        shock_thresh=1,
                        shock_lower_thresh=1000)

        n_cusp, bins_cusp = np.histogram(cusps,
                bins=cusp_range)
        cusp_data = (n_cusp, bins_cusp)
        n_shock, bins_shock = np.histogram(shocks,
                bins=shock_range)
        shock_data = (n_shock, bins_shock)

        try:
            np.save('out/{}cusp'.format(word[0]), cusp_data)
            np.save('out/{}shock'.format(word[0]), shock_data)
            print(f'did it for {word[0]}')
        except FileNotFoundError:
            print('not including {} for now because it is disgusting'.format(word[0]))
            continue

    sys.exit(0)

