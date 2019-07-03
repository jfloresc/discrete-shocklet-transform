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
            '--rotation',
            type=int,
            help='Element of R_4 to use. Default is 0 (id). Computed mod 4.'
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

    return parser.parse_known_args()


def _process(
        data,
        kernel,
        rotation,
        wmin,
        wmax,
        nw,
        kernel_args
        ):
    """Computes the shocklet (cusplet) transform on time series data.

    Computes the transform on each row of the passed data. The output will be of shape (data.shape[0], nw, data.shape[1]).
    """


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

    elif len(fnames) == 1:
        try:
            kernel_args = [float(x) for x in kernel_args]
        except Exception as e:
            print(f'When attempting to process kernel arguments error encountered:')
            print(f'{e}')
            sys.exit(1)

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

        _process(
                data,
                args.kernel,
                args.rotation,
                args.wmin,
                wmax,
                args.nw,
                kernel_args 
                )

    else:
        pass



if __name__ == "__main__":
    main()
