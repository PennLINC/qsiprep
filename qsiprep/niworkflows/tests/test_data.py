# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test data """

import os
from pathlib import Path
from qsiprep.niworkflows.data import utils


def test_get_data_path(monkeypatch):
    """ utils._get_data_path """
    # remove env variables
    monkeypatch.delitem(os.environ, 'CRN_SHARED_DATA', raising=False)
    monkeypatch.delitem(os.environ, 'CRN_DATA', raising=False)

    # pass explicit folders
    assert utils._get_data_path('/some/path') == [Path('/some/path')]
    assert utils._get_data_path('/some/path:/some/other') == \
        [Path('/some/path'), Path('/some/other')]

    # niworkflows default location
    assert utils._get_data_path() == [utils.NIWORKFLOWS_CACHE_DIR]

    monkeypatch.setenv('CRN_DATA', '~/.crn-data')
    # pass explicit folders
    assert utils._get_data_path('/some/path') == [Path('/some/path')]
    assert utils._get_data_path('/some/path:/some/other') == \
        [Path('/some/path'), Path('/some/other')]
    # niworkflows default location
    assert utils._get_data_path() == [Path('~/.crn-data').expanduser().resolve(),
                                      utils.NIWORKFLOWS_CACHE_DIR]

    monkeypatch.setenv('CRN_DATA', '~/.crn-data:/usr/local/share/crn-data')
    # niworkflows default location
    assert utils._get_data_path() == [Path('~/.crn-data').expanduser().resolve(),
                                      Path('/usr/local/share/crn-data'),
                                      utils.NIWORKFLOWS_CACHE_DIR]

    monkeypatch.setenv('CRN_DATA', '~/.crn-data:/usr/local/share/crn-data:')
    # niworkflows default location
    assert utils._get_data_path() == [Path('~/.crn-data').expanduser().resolve(),
                                      Path('/usr/local/share/crn-data'),
                                      utils.NIWORKFLOWS_CACHE_DIR]

    monkeypatch.setenv('CRN_DATA', '~/.crn-data')
    monkeypatch.setenv('CRN_SHARED_DATA', '/usr/local/share/crn-data:')
    # pass explicit folders
    assert utils._get_data_path('/some/path') == [Path('/some/path')]
    assert utils._get_data_path('/some/path:/some/other') == \
        [Path('/some/path'), Path('/some/other')]
    # niworkflows default location
    assert utils._get_data_path() == [Path('/usr/local/share/crn-data'),
                                      Path('~/.crn-data').expanduser().resolve(),
                                      utils.NIWORKFLOWS_CACHE_DIR]
