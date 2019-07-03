#!/usr/bin/env python

import sys
import os
import argparse
import multiprocessing
import pathlib

import joblib
import numpy as np

import cusplets


def parse_args():
    parser = argparse.ArgumentParser(
            description='Runs the STARS algorithm on data',
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
            default=','
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
            help='Kernel function to use. Must be in dir(cusplets)',
            default='power_cusp'
            )
    parser.add_argument(
            '-r',
            '--reflection',
            type=int,
            help='Element of R_4 to use. Default is 0 (id). Computed mod 4.',
            default=0
            )
    parser.add_argument(
            '-b',
            '--bvalue',
            type=float,
            help='Multiplier for std dev in classification',
            default=0.75
            )
    parser.add_argument(
            '-g',
            '--geval',
            type=float,
            help='Threshold for window construction',
            default=0.5
            )
    parser.add_argument(
            '-l',
            '--lookback',
            type=int,
            help='Number of indices to look back for window construction',
            default=0
            )
    parser.add_argument(
            '-w',
            '--weighting',
            type=str,
            help='Method for weighting of cusp indicator functions. Must be in dir(cusplets)',
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
            help='Largest kernel size. Defaults to min {500, 1/2 length of time series}.',
            default=500
            )
    parser.add_argument(
            '-nw',
            '--nw',
            type=int,
            help='Number of kernels to use. Ideally (wmax - wmin) / nw would be an integer.',
            default=100
            )
    parser.add_argument(
            '-s',
            '--savespec',
            type=str,
            help='Spec for saving. Options are: cc, to just save cusplet transform; '
            'indic, to save indicator function; all, to save everything',
            default='all'
            )
    parser.add_argument(
            '-norm',
            '--norm',
            type=bool,
            help='Whether or not to normalize series to be wide-sense stationary '
            'with intertemporal zero mean and unit variance',
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
        
        cc, _ = cusplets.cusplet( 
                    row,
                    kernel,
                    widths,
                    k_args=kernel_args,
                    reflection=reflection
                )
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
                    cc=cc,
                    indic=sum_cc )
            else:
                windows = cusplets.make_components(
                        gearray,
                        scan_back=nback
                        )
                weighted_sumcc = np.copy(sum_cc)

                for window in windows:
                    weighted_sumcc[window] *= weighting( row[window] )

                np.savez_compressed( outdir + '/' + f'{orig_fname}-row{i}',
                    cc=cc,
                    indic=sum_cc,
                    windows=windows,
                    weighted_indic=weighted_sumcc )


def _mp_process( fname, args, kernel_args ):
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



if __name__ == "__main__":
    main()
