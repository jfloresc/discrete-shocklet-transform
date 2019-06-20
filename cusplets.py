#!/usr/bin/env python

import numpy as np
from scipy import signal


def zero_norm(arr):
    arr = 2 * (arr - min(arr)) / (max(arr) - min(arr)) - 1
    return arr - np.sum(arr) / len(arr)


def make_prob(arr):
    arr = arr - min(arr)
    arr = arr / np.sum(arr)
    return arr


def normalize(arr, stats=False):
    arr = np.array(arr)
    mean = arr.mean()
    std = arr.std()
    normed = (arr - mean) / std
    if not stats:
        return normed
    return normed, mean, std 


def renormalize(arr, mean, std):
    arr = np.array(arr)
    return (arr - mean) / std


def haar(L, zn=True):
    res = -1 * np.ones(L)
    res[len(res) // 2:] = 1
    if zn:
        return zero_norm(res)
    return res


def power_cusp(L, b, zn=True):
    res = power_zero_cusp(L, b, zn=zn) + power_zero_cusp(L, b, zn=zn)[::-1]
    if zn:
        return zero_norm(res)
    return res


def pitchfork(L, b, zn=True):
    res = power_zero_cusp(L, 2*b, zn=zn)[::-1] + power_cusp(L, b, zn=zn) + power_zero_cusp(L, 2*b, zn=zn)
    if zn:
        return zero_norm(res)
    return res


def power_zero_cusp(L, b, zn=True):
    x = np.linspace(1, 4, L)
    res = x**b
    res[len(res) // 2:] = 0
    if zn:
        return zero_norm(res)
    return res


def exp_cusp(L, a, zn=True):
    res = exp_zero_cusp(L, a, zn=zn) + exp_zero_cusp(L, a, zn=zn)[::-1]
    if zn:
        return zero_norm(res)
    return res


def exp_zero_cusp(L, a, zn=True):
    x = np.linspace(1, 4, L)
    res = np.exp(a * x)
    res[len(res) // 2:] = 0
    if zn:
        return zero_norm(res)
    return res


def cusplet(arr, kernel, widths, k_args=[], reflection=0, width_weights=None):
    """
    Implements the discrete cusplet transform.

    :param arr: array of shape (n,) or (n,1). This array should not contain inf-like values.
    The transform will still be computed but infs propagate. Nan-like values will be linearly 
    interpolated, which is okay for subsequent time-based analysis but will introduce ringing 
    in frequency-based analyses.
    :type arr: list, tuple, or numpy.ndarray

    :param kernel: kernel function. Must take an integer L > 0 as an 
    argument and any number of additional, nonkeyword arguments, 
    and returns a numpy array of shape (L,) that implements the kernel. 
    The returned array should sum to zero; use the zero_norm function for this.
    :type kernel: callable

    :param widths: iterable of integers that specify the window widths (L above).
    Assumed to be in increasing order; if widths is not in increasing order the 
    results will be garbage.
    :type widths: iterable

    :param k_args: arguments for the kernel function.
    :type k_args: list or tuple

    :param reflection: integer n evaluates to n %4, element of the reflection group
    that left-multiplies the kernel function. Default is 0 (identity element).
    :type reflection: int

    :return tuple -- (numpy array of shape (L, n) -- the cusplet transform,
    k -- the calculated kernel function)
    """
    arr = np.array( arr )
    cc = np.zeros((len(widths), len(arr)))

    # we need to fill all nans
    # best choice for this is linear interpolation

    nans = np.isnan(arr)
    nonzero_fn = lambda x: x.nonzero()[0]

    arr[nans] = np.interp(nonzero_fn(nans),
            nonzero_fn(~nans),
            arr[~nans])

    # allow for weighting of width importance
    if width_weights is None:
        width_weights = np.ones_like(widths)
   
    # now we will see what group action to operate with
    for i, w in enumerate(widths):
        k = kernel(w, *k_args)
        # implement reflections
        reflection = reflection % 4
        if reflection == 1:
            k = k[::-1]
        elif reflection == 2:
            k = -k
        elif reflection == 3:
            k = -k[::-1]

        cc[i] = width_weights[i] * np.correlate(arr, k, mode='same')
    return cc, k


def cusplet_parameter_sweep(arr, kernel, widths, k_args, reflection=0, width_weights=None, k_weights=None):
    """
    Sweeps over values of parameters (kernel arguments) in the discrete cusplet transform.

    :param arr: numpy array of shape (n,) or (n,1), time series
    :type arr: list, tuple, or numpy.ndarray

    :param kernel: kernel function. Must take an integer L > 0 as an 
    argument and any number of additional, nonkeyword arguments, 
    and returns a numpy array of shape (L,) that implements the kernel. 
    The returned array should sum to zero; use the zero_norm function for this.
    :type kernel: callable

    :param widths: iterable of integers that specify the window widths (L above).
    Assumed to be in increasing order; if widths is not in increasing order the 
    results will be garbage.
    :type widths: iterable

    :param k_args: iterable of iterables of arguments for the kernel function.
    Each top-level iterable is treated as a single parameter vector.
    :type k_args: list or tuple of lists or tuples

    :param reflection: integer n evaluates to n %4, element of the reflection group
    that left-multiplies the kernel function. Default is 0 (identity element).
    :type reflection: int

    :return numpy.ndarray -- numpy array of shape (L, n, len(k_args)), the cusplet transform
    """
    k_args = np.array(k_args)

    if k_weights is None:
        k_weights = np.ones(k_args.shape[0])
    
    cc = np.zeros((len(widths), len(arr), len(k_args)))
    
    for i, k_arg in enumerate(k_args):
        cres, _ = cusplet(arr,
                kernel,
                widths,
                k_args=k_arg,
                reflection=reflection,
                width_weights=width_weights)
        cc[:, :, i] = cres * k_weights[i]

    return cc 
    

def classify_cusps(cc, b=1, geval=False):
    """
    Classifies points as belonging to cusps or not.

    :param cc: numpy array of shape (L, n), the cusplet transform of 
    a time series
    :type cc: numpy.ndarray

    :param b: multiplier of the standard deviation.
    :type b: int or float

    :param geval: optional. If geval is an int or float, classify_cusps will return
    (in additoin to the cusps and cusp intensity function) an array of points where the cusp intensity 
    function is greater than geval.

    :return tuple --- (numpy.ndarray of indices of the cusps;
    numpy.ndarray representing the cusp intensity function) 
    or, if geval is not False, (extrema; the cusp intensity function; array of points where 
    the cusp intensity function is greater than geval)
    """
    sum_cc = zero_norm(np.nansum(cc, axis=0))
    mu_cc = np.nanmean(sum_cc)
    std_cc = np.nanstd(sum_cc)

    extrema = np.array( signal.argrelextrema(sum_cc, np.greater) )[0]
    extrema = [x for x in extrema if sum_cc[x] > mu_cc + b * std_cc]

    if geval is False:
        return extrema, sum_cc

    gez = np.where(sum_cc > geval)
    return extrema, sum_cc, gez


def _make_components(indicator, cusp_points=None):
    """
    Get individual windows from array of indicator indices.

    Takes cusp indicator function and returns windows of contiguous cusp-like behavior. 
    If an array of hypothesized deterministic peaks of cusp-like behavior is passed, 
    thins these points so that there is at most one point per window.

    :param indicator: array of the points where the cusp intensity function exceeds some threshold
    :type indicator: list

    :param cusp_points: optional, array of points that denote the hypothesized deterministic peaks of 
    cusps
    :type cusp_points: list or numpy.ndarray

    :return list -- the contiguous cusp windows; or, if cusp_points is not None, tuple --
    (the contiguous cusp windows, the thinned cusp points)
    """
    windows = []
    indicator = np.array(indicator)
    if len(indicator.shape) > 1:
        indicator = indicator[0]
    j = 0
    
    for i, x in enumerate(indicator):
        if i == len(indicator) - 1:
            window = indicator[j : i]
            if len(window) >= 2:
                windows.append(window)
            break
        elif indicator[i + 1] == x + 1:
            continue  # still part of the same block
        else:  # block has ended
            window = indicator[j : i]
            if len(window) >= 2:
                windows.append(window)
            j = i + 1

    if cusp_points is None:
        return windows

    pt_holder = [[] for _ in range(len(windows))]

    for pt in cusp_points:
        for i, window in enumerate(windows):
            if (pt >= window[0]) and (pt <= window[-1]):
                pt_holder[i].append(pt)
                break

    windows_ = []
    estimated_cusp_points = []

    for holder, window in zip(pt_holder, windows):
        if holder != []:
            windows_.append (window )
            estimated_cusp_points.append(int(np.median(holder)))

    estimated_cusp_points = np.array(estimated_cusp_points, dtype=int)

    return windows_, estimated_cusp_points


def make_components(indicator, cusp_points=None, scan_back=0):
    """
    Get individual windows from array of indicator indices.

    Takes cusp indicator function and returns windows of contiguous cusp-like behavior. 
    If an array of hypothesized deterministic peaks of cusp-like behavior is passed, 
    thins these points so that there is at most one point per window.
    The scan_back parameter connects contiguous windows if they are less than or equal to 
    scan_back indices from each other.

    :param indicator: array of the points where the cusp intensity function exceeds some threshold
    :type indicator: list

    :param cusp_points: optional, array of points that denote the hypothesized deterministic peaks of 
    cusps
    :type cusp_points: list or numpy.ndarray

    :param scan_back: number of indices to look back. If cusp windows are within scan_back indices
    of each other, they will be connected into one contiguous window.
    :type scan_back: int >= 0

    :return list -- the contiguous cusp windows; or, if cusp_points is not None, tuple --
    (the contiguous cusp windows, the thinned cusp points)
    """
    windows = _make_components(indicator, cusp_points=cusp_points)

    if cusp_points is not None:
        windows, estimated_cusp_points = windows

    if (len(windows) > 1) and (scan_back > 0):
        windows_ = []
        for i in range(len(windows)):
            if len(windows_) == 0:
                windows_.append( list(windows[i]) )
            else:
                if windows[i][0] <= windows_[-1][-1] + scan_back:
                    fill_between = list( range(windows_[-1][-1] + 1,
                        windows[i][0]) )
                    windows_[-1].extend( fill_between )
                    windows_[-1].extend( list(windows[i]) )
                else:
                    windows_.append( list(windows[i]) )
    
    else:
        windows_ = windows
    
    if cusp_points is None:
        return windows_
    return windows_, estimated_cusp_points


def window_argmaxes(windows, data):
    """
    Find argmax point in data within each window.

    :param windows: a list of indices indicating targeted windows in the data array
    :type windows: a 2D list or 2D numpy.ndarray

    :param data: a list of data points
    :type data: a list or numpy.ndarray

    :return numpy.array -- max points for each window.
    """
    data = np.array(data)
    argmaxes = []

    for window in windows:
        data_segment = data[window]
        argmaxes.append(window[np.argmax(data_segment)])

    return np.array(argmaxes)


def max_change(arr):
    """
    Calculates the difference between the max and min points in an array.

    :param arr: a time series for a given word
    :type arr: a list or numpy.ndarray

    :return float -- maximum relative change
    """
    return np.max(arr) - np.min(arr)


def max_rel_change(arr, neg=True):
    """
    Calculates the maximum relative changes in an array (log10).

    One possible choice for a weighting function in cusplet transform.

    :param arr: a time series for a given word
    :type arr: a list or numpy.ndarray

    :param neg: (if true) arr - np.min(arr) + 1

    :return  float -- maximum relative change (log10)
    """
    if neg:
        arr = arr - np.min(arr) + 1

    logret = np.diff(np.log10(arr))
    return np.max(logret) - np.min(logret)


def top_k(indices, words, k):
    """
    Find the top k words by the weighted cusp indicator function

    :param indices: a list of indicator values
    :type indices: list or numpy.array

    :param words: a list of strings
    :type words: list or numpy.array

    :param k: number of indices to look up.
    :type k: int > 0

    :return list -- length k list of (word, indicator value) tuples
    """
    inds = np.argpartition(indices, -k)[-k:]
    topkwords = words[inds]
    topkvals = indices[inds]
    top = [(word, val) for word, val in zip(topkwords, topkvals)]
    top = sorted(top, key=lambda t: t[1], reverse=True)
    return top
