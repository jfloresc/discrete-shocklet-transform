#!/usr/bin/env python

import sys
import os
import argparse
import multiprocessing
import pathlib

import numpy as np

import cusplets


def parse_args():
    parser = argparse.ArgumentParser(
            description='Runs the Shocklet Transform And Ranking (STAR) algorithm on data',
            formatter_class=argparse.RawTextHelpFormatter
            )
    parser.add_argument(
            '-i',
            '--input',
            type=str,
            help='Path to files on which to run the algorithm. '
            'This file should be in row-major order. That is, '
            'it should be a N_variable x T matrix, where T is '
            'the number of timesteps.',
            default='.'
            )
    parser.add_argument(
            '-e',
            '--ending',
            type=str,
            help='Ending of files on which to run algorithm. Must be readable to numpy.genfromtxt()',
            default='csv'
            )
    parser.add_argument(
            '-d',
            '--delimiter',
            type=str,
            help='Delimiter of entries in files',
            default='none'
            )
    parser.add_argument(
            '-o',
            '--output',
            type=str,
            help='Path to which to save output',
            default='./out'
            )
    parser.add_argument(
            '-k',
            '--kernel',
            type=str,
            help='Kernel function to use. Must be in dir(cusplets). Pre-written options (no '
            'need to write code) are: haar, which is the Haar wavelet and looks for pure level '
            'changes; power_zero_cusp, which is the building block for other cusp kernels and '
            'looks for power (monomial) growth followed by an abrupt drop to constant low levels; '
            'power_cusp, which is a power cusp shape; exp_zero_cusp, which is like '
            'power_zero_cusp except with exponential growth; and exp_cusp, which is like '
            'power_cusp except with exponential growth. Combining these kernels with a reflection '
            '(computed using the option `-r <int>`) is probably enough to look for any interesting '
            'behavior. If you want to write your own kernel function, it must be in cusplets.py and '
            'conform to the API `function(W, *args, zn=True)` where W is a window size, '
            '*args are any remaining positional arguments to the function as parameters and must '
            'be cast-able to `float`, and zn is a boolean corresponding to whether or not to '
            'ensure that the kernel function integrates to zero, which it should by default. '
            'Kernel defaults to power_cusp.',
            default='power_cusp'
            )
    parser.add_argument(
            '-r',
            '--reflection',
            type=int,
            help='Element of the reflection group R_4 to use. Default is 0 (id). Computed mod 4.',
            default=0
            )
    parser.add_argument(
            '-b',
            '--bvalue',
            type=float,
            help='Multiplier for std dev in classification. Default is b=0.75.',
            default=0.75
            )
    parser.add_argument(
            '-g',
            '--geval',
            type=float,
            help='Threshold for window construction. Default is 0.5.',
            default=0.5
            )
    parser.add_argument(
            '-l',
            '--lookback',
            type=int,
            help='Number of indices to look back for window construction. Default is 0.',
            default=0
            )
    parser.add_argument(
            '-w',
            '--weighting',
            type=str,
            help='Method for weighting of cusp indicator functions. Must be in dir(cusplets). '
            'Defaults to max_change, the maximum minus the minimum value of original series '
            'within each window. The other pre-written option (no need to write code) is '
            'max_rel_change, which computes max_change on the array of log returns of the '
            'original time series. If you want to write your own weighting function, it must '
            'be in cusplets.py and correspond to the API `function(arr)` where arr is an array.',
            default='max_change'
            )
    parser.add_argument(
            '-wmin',
            '--wmin',
            type=int,
            help='Smallest kernel size. Defaults to 10.',
            default=10
            )
    parser.add_argument(
            '-wmax',
            '--wmax',
            type=int,
            help='Largest kernel size. Defaults to min{500, 1/2 length of time series}.',
            default=500
            )
    parser.add_argument(
            '-nw',
            '--nw',
            type=int,
            help='Number of kernels to use. Ideally (wmax - wmin) / nw would be an integer. ' 
            'Default is 100.',
            default=100
            )
    parser.add_argument(
            '-s',
            '--savespec',
            type=str,
            help='Spec for saving. Options are: cc, to just save cusplet transform; '
            'indic, to save indicator function; windows, to save anomalous windows; '
            'weighted, to save weighted indicator function; '
            'all, to save everything. Defaults to all. Files are saved in compressed '
            '.npz numpy archive format.',
            default='all'
            )
    parser.add_argument(
            '-norm',
            '--norm',
            type=bool,
            help='Whether or not to normalize series to be wide-sense stationary '
            'with intertemporal zero mean and unit variance. Default is False.',
            default=False
            )

    return parser.parse_known_args()


def _process(
        data,
        kernel,
        reflection,
        wmin,
        wmax,
        nw,
        kernel_args,
        outdir,
        weighting,
        savespec,
        b,
        geval,
        nback,
        orig_fname,
        norm
        ):
    """Computes the shocklet (cusplet) transform on time series data.

    Computes the transform on each row of the passed data. The collection of cusplet transforms will be of shape (data.shape[0], nw, data.shape[1]).
    """
    kernel = getattr(cusplets, kernel)
    weighting = getattr(cusplets, weighting)
    widths = np.linspace(wmin, wmax, nw).astype(int)

    for i, row in enumerate( data ):
        if norm:
            row = cusplets.normalize(row)
        try: 
            cc, _ = cusplets.cusplet( 
                    row,
                    kernel,
                    widths,
                    k_args=kernel_args,
                    reflection=reflection
                    )
        except Exception as e:
            print(f'Error occurred in computation of shocklet transform of {orig_fname}')
            print(f'Error: {e}')
            return

        if savespec == 'cc':
            np.savez_compressed( outdir + f'{orig_fname}-row{i}',
                    cc=cc )
        else:
            extrema, sum_cc, gearray = cusplets.classify_cusps( 
                    cc,
                    b=b,
                    geval=geval
                    )
            if savespec == 'indic':
                np.savez_compressed( outdir + f'{orig_fname}-row{i}',
                    indic=sum_cc )
            elif savespec in ['windows', 'weighted', 'all']:
                windows = cusplets.make_components(
                        gearray,
                        scan_back=nback
                        )
                if savespec == 'windows':
                    np.savez_compressed( outdir + f'{orig_fname}-row{i}',
                        windows=windows )
                
                else:
                    weighted_sumcc = np.copy(sum_cc)

                    for window in windows:
                        weighted_sumcc[window] *= weighting( row[window] )

                    if savespec == 'weighted':
                        np.savez_compressed( outdir + f'{orig_fname}-row{i}',
                            weighted_indic=weighted_sumcc )
                    else:
                        np.savez_compressed( outdir + '/' + f'{orig_fname}-row{i}',
                            cc=cc,
                            indic=sum_cc,
                            windows=windows,
                            weighted_indic=weighted_sumcc )


def _mp_process( fname, args, kernel_args ):
    if args.delimiter == 'none':
        data = np.genfromtxt( fname )
    else:
        data = np.genfromtxt( fname, delimiter=args.delimiter )
    if len(data.shape) < 2:
        data = data.reshape(1, data.shape[0])
    # fix up the window size if we need to
    # first check to see if they are compatible
    if args.wmin >= args.wmax:
        print(f'wmin must be less than wmax for sensible output.')
        return
    nt = data.shape[1]
    wmax = min( args.nw, int(0.5 * nt) )

    # ensure we have a valid savespec
    if args.savespec not in ['cc', 'indic', 'all']:
        print(f'{args.savespec} not one of cc, indic, or all. Defaulting to all.')
        savespec = 'all'
    else:
        savespec = args.savespec

    _process(
            data,
            args.kernel,
            args.reflection,
            args.wmin,
            wmax,
            args.nw,
            kernel_args,
            args.output,
            args.weighting,
            savespec,
            args.bvalue,
            args.geval,
            args.lookback,
            fname.split('/')[-1].split('.')[0],
            args.norm
            )



def main():
    args, kernel_args = parse_args()
    # first look for all files that match the requested
    if not os.path.isdir( args.input ):
        print(f'{args.input} does not exist or is not a directory')
        sys.exit(1)
    else:
        fnames = [args.input + '/' + f for f in os.listdir(args.input) if f.endswith(args.ending)]
    
    outpath = pathlib.Path(args.output).mkdir(
            exist_ok=True,
            parents=True
            )

    if len(fnames) == 0:
        print(f'There are no files with ending {args.ending} in {args.input}')
        sys.exit(1)

    elif len(fnames) >= 1:
        try:
            kernel_args = [float(x) for x in kernel_args]
        except Exception as e:
            print(f'When attempting to process kernel arguments error encountered:')
            print(f'{e}')
            sys.exit(1)

        if len(fnames) == 1:
            data = np.genfromtxt( fnames[0], delimiter=args.delimiter )
            if len(data.shape) < 2:
                data = data.reshape(1, data.shape[0])

            # fix up the window size if we need to
            # first check to see if they are compatible
            if args.wmin >= args.wmax:
                print(f'wmin must be less than wmax for sensible output.')
                sys.exit(1)
            nt = data.shape[1]
            wmax = min( args.nw, int(0.5 * nt) )

            # ensure we have a valid savespec
            if args.savespec not in ['cc', 'indic', 'all']:
                print(f'{args.savespec} not one of cc, indic, or all. Defaulting to all.')
                savespec = 'all'
            else:
                savespec = args.savespec

            _process(
                    data,
                    args.kernel,
                    args.reflection,
                    args.wmin,
                    wmax,
                    args.nw,
                    kernel_args,
                    args.output,
                    args.weighting,
                    savespec,
                    args.bvalue,
                    args.geval,
                    args.lookback,
                    fnames[0].split('/')[-1].split('.')[0],
                    args.norm
                    )

        else:  # multiprocess over files
            pool = multiprocessing.Pool(None)
            errors = pool.starmap( 
                    _mp_process,
                    zip( 
                        fnames,
                        (args for _ in range(len(fnames))),
                        (kernel_args for _ in range(len(fnames)))
                        )
                    )
            sys.exit()


if __name__ == "__main__":
    main()
