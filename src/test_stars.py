#!/usr/bin/env python

import subprocess

import pytest


def test_no_input_dir_raises():
    with pytest.raises(subprocess.CalledProcessError) as pytest_wrapped_e:
        call = subprocess.check_output( [
                './stars.py',
                '-i',
                'DNE_DIR'
                ] )
    assert pytest_wrapped_e.type == subprocess.CalledProcessError


def test_no_input_dir_text():
    try:
        call = subprocess.check_output( [
            './stars.py',
            '-i',
            'DNE_DIR'
            ] )
    except subprocess.CalledProcessError as e:
        assert e.output == b'DNE_DIR does not exist or is not a directory\n'
