#!/usr/bin/env python3
"""Download test data."""

import sys

from qsiprep.tests.utils import download_test_data

if __name__ == '__main__':
    data_dir = sys.argv[1]
    dset = sys.argv[2]
    download_test_data(dset, data_dir)
