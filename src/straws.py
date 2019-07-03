#!/usr/bin/env python

import argparse
import os
import multiprocessing
import sys

import numpy as np
import pandas as pd

import cusplets


_ext2rname = {'csv' : 'read_csv',
        'json' : 'read_json',
        'xls' : 'read_excel', 
        'xlsx' : 'read_excel',
        'parq' : 'read_parquet',
        'sql' : 'read_sql',
        'xpt' : 'read_sas', 
        'sas7bdat' : 'read_sas',
        'pkl' : 'read_pickle',
        'feather' : 'read_feather'}


def parse_args():
    parser = argparse.ArgumentParser(
            description='Run STRAWS algorithm on data',
            formatter_class=argparse.RawTextHelpFormatter
            )
    
    parser.add_argument(
            '-i',
            '--input',
            type=str,
            help='''
            Path (relative or absolute) to input file or directory.
            If a directory, the algorithm will be run on each file in the directory.

            Files can be one of two things: either a .npy numpy archive or 
            (preferred) something coercible to a pandas dataframe (will be loaded using 
            `pd.read_{extension}(filename)` where `extension` is a string trailing the last `.`
            in the filename. Supported extensions are csv, json, xls, xlsx, parq, sql, 
            xpt, sas7bdat, pkl, and feather. 
            If there is no `.` in the filename, loading will be attempted 
            using `pd.read_csv(filename)`.)
            
            Data should follow row-major order; each row should be a unique different time
            series.
            ''',
            required=True
            )

    parser.add_argument(
            '-o',
            '--output',
            type=str,
            help='Path to output file',
            required=True
            )

    parser.add_argument( 
            '-ck',
            '--customkernels',
            action='count',
            help='Whether or not to use custom kernel files'
            )

    parser.add_argument(
            '-ckp',
            '--customkernelpath',
            type=str,
            nargs='?',
            help='Path to custom kernel files'
            )
    parser.add_argument(
            '-mp',
            '--multiproc',
            action='count',
            help='Whether to use multiprocessing. Default is to single process.'
            )
    parser.add_argument(
            '-mps',
            '--multiprocsetup',
            type=str,
            nargs='?',
            default='files',
            help='''
            Multiprocessing scheme.

            If multiprocsetup == 'array', workers will handle rows of arrays
            and arrays will be iterated over. If multiprocsetup == 'files' (default), 
            each worker will handle one file at a time.
            '''
            )
    parser.add_argument(
            '-cores', 
            '--cores',
            type=int,
            nargs='?',
            default=-1,
            help='Number of cores to use for multiprocess. Default is -1 (all cores)'
            )

    return parser.parse_args()


def _load_file( fname ):
    fname_l = fname.split('.')

    if len(fname_l) == 1:
        try:
            return pd.read_csv( fname )
        except Exception as e:
            print(f'Cannot load {fname}: {e}')
            return

    else:
        extension = fname_l[-1]
        if extension == 'npy':
            return np.load( fname )
        else:
            try:
                return getattr(pd, _ext2rname[extension])( fname )
            except Exception as e:
                print(f'Cannot load {fname}: {e}')
                return



def _process( fname, mp=False, cores=-1, out='.' ):
    """Run STRAWS algorithm on data located on disk at fname
    """
    # data can be one of two things
    # either can be numpy array saved as .npy or columnar-style format
    data = _load_file( fname )

    if mp is True:
        pool = multiprocessing.Pool( cores ) 
    else:
        pass


def _process_from_directory( directory, mp=False, how='files', cores=-1, out='.' ):
    fnames = [directory + '/' + x for x in os.listdir( directory )]
    outs = itertools.repeat(out, len(fnames))

    if (mp is False) or (how == 'array'):
        for fname in fnames:
            _process( fname )
    else:
        pool = multiprocessing.Pool( cores )
        errors = pool.starmap( 
                _process,
                zip(fnames, outs)
                )



def main():
    args = parse_args()

    # do we have one input file or many?
    if os.path.isdir( args.input ):
        _process_from_directory( 
                args.input, 
                mp=args.multiproc,
                how=args.multiprocsetup,
                cores=args.cores,
                out=args.output
                )
    else:
        pass






if __name__ == "__main__":
    main()
