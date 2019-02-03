#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for data grabbers (from nilearn)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path as op
from pathlib import Path
import sys
import shutil
import time
import base64
import hashlib
import subprocess as sp
from builtins import str

try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError, URLError

from .. import NIWORKFLOWS_LOG

PY3 = sys.version_info[0] > 2
MAX_RETRIES = 20
NIWORKFLOWS_CACHE_DIR = Path.home() / '.cache' / 'stanford-crn'


def fetch_file(dataset_name, url, dataset_dir, dataset_prefix=None,
               default_paths=None, filetype=None, resume=True, overwrite=False,
               md5sum=None, username=None, password=None, retry=0,
               verbose=1, temp_downloads=None):
    """Load requested file, downloading it if needed or requested.

    :param str url: contains the url of the file to be downloaded.
    :param str dataset_dir: path of the data directory. Used for data
        storage in the specified location.
    :param bool resume: if true, try to resume partially downloaded files
    :param overwrite: if bool true and file already exists, delete it.
    :param str md5sum: MD5 sum of the file. Checked if download of the file
        is required
    :param str username: username used for basic HTTP authentication
    :param str password: password used for basic HTTP authentication
    :param int verbose: verbosity level (0 means no message).
    :returns: absolute path of downloaded file
    :rtype: str

    ..note::

      If, for any reason, the download procedure fails, all downloaded files are
      removed.


    """
    final_path, cached = _get_dataset(dataset_name, dataset_prefix=dataset_prefix,
                                      data_dir=dataset_dir, default_paths=default_paths,
                                      verbose=verbose)
    if cached and not overwrite:
        return final_path

    data_dir = final_path.parent

    if temp_downloads is None:
        temp_downloads = NIWORKFLOWS_CACHE_DIR / 'downloads'
    temp_downloads = Path(temp_downloads)

    temp_downloads.mkdir(parents=True, exist_ok=True)

    # Determine filename using URL
    parse = urlparse(url)
    file_name = op.basename(parse.path)
    if file_name == '':
        file_name = _md5_hash(parse.path)

        if filetype is not None:
            file_name += filetype

    temp_full_path = temp_downloads / file_name
    temp_part_path = temp_full_path.with_name(file_name + '.part')

    if overwrite:
        shutil.rmtree(str(dataset_dir), ignore_errors=True)

        if temp_full_path.exists():
            temp_full_path.unlink()

    t_0 = time.time()
    local_file = None
    initial_size = 0

    # Download data
    request = Request(url)
    request.add_header('Connection', 'Keep-Alive')
    if username is not None and password is not None:
        if not url.startswith('https'):
            raise ValueError(
                'Authentication was requested on a non  secured URL ({0!s}).'
                'Request has been blocked for security reasons.'.format(url))
        # Note: HTTPBasicAuthHandler is not fitted here because it relies
        # on the fact that the server will return a 401 error with proper
        # www-authentication header, which is not the case of most
        # servers.
        encoded_auth = base64.b64encode(
            (username + ':' + password).encode())
        request.add_header(b'Authorization', b'Basic ' + encoded_auth)
    if verbose > 0:
        displayed_url = url.split('?')[0] if verbose == 1 else url
        NIWORKFLOWS_LOG.info('Downloading data from %s ...', displayed_url)
    if resume and temp_part_path.exists():
        # Download has been interrupted, we try to resume it.
        local_file_size = temp_part_path.stat().st_size
        # If the file exists, then only download the remainder
        request.add_header("Range", "bytes={}-".format(local_file_size))
        try:
            data = urlopen(request)
            content_range = data.info().get('Content-Range')
            if (content_range is None or not content_range.startswith(
                    'bytes {}-'.format(local_file_size))):
                raise IOError('Server does not support resuming')
        except Exception:
            # A wide number of errors can be raised here. HTTPError,
            # URLError... I prefer to catch them all and rerun without
            # resuming.
            if verbose > 0:
                NIWORKFLOWS_LOG.warn(
                    'Resuming failed, try to download the whole file.')
            return fetch_file(
                dataset_name, url, dataset_dir,
                resume=False, overwrite=overwrite,
                md5sum=md5sum, username=username, password=password,
                verbose=verbose)
        initial_size = local_file_size
        mode = 'ab'
    else:
        try:
            data = urlopen(request)
        except (HTTPError, URLError):
            if retry < MAX_RETRIES:
                if verbose > 0:
                    NIWORKFLOWS_LOG.warn('Download failed, retrying (attempt %d)',
                                         retry + 1)
                time.sleep(5)
                return fetch_file(
                    dataset_name, url, dataset_dir, resume=False, overwrite=overwrite,
                    md5sum=md5sum, username=username, password=password,
                    verbose=verbose, retry=retry + 1)
            else:
                raise
        mode = 'wb'

    with temp_part_path.open(mode) as local_file:
        _chunk_read_(data, local_file, report_hook=(verbose > 0),
                     initial_size=initial_size, verbose=verbose)
    temp_part_path.replace(temp_full_path)
    delta_t = time.time() - t_0
    if verbose > 0:
        # Complete the reporting hook
        sys.stderr.write(' ...done. ({0:.0f} seconds, {1:.0f} min)\n'
                         .format(delta_t, delta_t // 60))

    if md5sum is not None:
        if _md5_sum_file(temp_full_path) != md5sum:
            raise ValueError("File {!s} checksum verification has failed."
                             " Dataset fetching aborted.".format(temp_full_path))

    if filetype is None:
        fname, filetype = op.splitext(temp_full_path.name)
        if filetype == '.gz':
            fname, ext = op.splitext(fname)
            filetype = ext + filetype

    if filetype.startswith('.'):
        filetype = filetype[1:]

    if filetype.startswith('tar'):
        args = 'xf' if not filetype.endswith('gz') else 'xzf'
        sp.check_call(['tar', args, str(temp_full_path)], cwd=data_dir)
        temp_full_path.unlink()
        return final_path

    if filetype == 'zip':
        import zipfile
        sys.stderr.write('Unzipping package (%s) to data path (%s)...' % (
            temp_full_path, data_dir))
        with zipfile.ZipFile(str(temp_full_path), 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        sys.stderr.write('done.\n')
        return final_path

    return final_path


def _get_data_path(data_dir=None):
    """ Get data storage directory

    data_dir: str
      Path of the data directory. Used to force data storage in
      a specified location.

    :returns:
        a list of paths where the dataset could be stored,
        ordered by priority

    """
    data_dir = data_dir or ''

    default_dirs = [Path(d).expanduser().resolve()
                    for d in os.getenv('CRN_SHARED_DATA', '').split(os.pathsep)
                    if d.strip()]
    default_dirs += [Path(d).expanduser().resolve()
                     for d in os.getenv('CRN_DATA', '').split(os.pathsep)
                     if d.strip()]
    default_dirs += [NIWORKFLOWS_CACHE_DIR]

    return [Path(d).expanduser()
            for d in data_dir.split(os.pathsep) if d.strip()] or default_dirs


def _get_dataset(dataset_name, dataset_prefix=None,
                 data_dir=None, default_paths=None,
                 verbose=1):
    """ Create if necessary and returns data directory of given dataset.

    data_dir: str
      Path of the data directory. Used to force data storage in
      a specified location.
    default_paths: list(str)
      Default system paths in which the dataset may already have been installed
      by a third party software. They will be checked first.
    verbose: int
      verbosity level (0 means no message).

    :returns: the path of the given dataset directory.
    :rtype: str

    .. note::

      This function retrieves the datasets directory (or data directory) using
      the following priority :

        1. defaults system paths
        2. the keyword argument data_dir
        3. the global environment variable CRN_SHARED_DATA
        4. the user environment variable CRN_DATA
        5. ~/.cache/stanford-crn in the user home folder


    """

    dataset_folder = dataset_name if not dataset_prefix \
        else '%s%s' % (dataset_prefix, dataset_name)
    default_paths = default_paths or ''
    paths = [p / dataset_folder for p in _get_data_path(data_dir)]
    all_paths = [Path(p) / dataset_folder
                 for p in default_paths.split(os.pathsep)] + paths

    # Check if the dataset folder exists somewhere and is not empty
    for path in all_paths:
        if path.is_dir() and list(path.iterdir()):
            if verbose > 1:
                NIWORKFLOWS_LOG.info(
                    'Dataset "%s" already cached in %s', dataset_name, path)
            return path, True

    for path in paths:
        if verbose > 0:
            NIWORKFLOWS_LOG.info(
                'Dataset "%s" not cached, downloading to %s', dataset_name, path)
        path.mkdir(parents=True, exist_ok=True)
        return path, False


def readlinkabs(link):
    """
    Return an absolute path for the destination
    of a symlink
    """
    path = os.readlink(link)
    if op.isabs(path):
        return path
    return op.join(op.dirname(link), path)


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
    """
    with Path(path).open('rb') as fhandle:
        md5sum = hashlib.md5()
        while True:
            data = fhandle.read(8192)
            if not data:
                break
            md5sum.update(data)
    return md5sum.hexdigest()


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=1):
    """Download a file chunk by chunk and show advancement

    :param urllib.response.addinfourl response: response to the download
        request in order to get file size
    :param str local_file: hard disk file where data should be written
    :param int chunk_size: size of downloaded chunks. Default: 8192
    :param bool report_hook: whether or not to show downloading advancement
    :param int initial_size: if resuming, indicate the initial size of the file
    :param int total_size: Expected final size of download (None means it
        is unknown).
    :param int verbose: verbosity level (0 means no message).
    :returns: the downloaded file path.
    :rtype: string

    """
    try:
        if total_size is None:
            total_size = response.info().get('Content-Length').strip()
        total_size = int(total_size) + initial_size
    except Exception as exc:
        if verbose > 2:
            NIWORKFLOWS_LOG.warn('Total size of chunk could not be determined')
            if verbose > 3:
                NIWORKFLOWS_LOG.warn("Full stack trace: %s", str(exc))
        total_size = None
    bytes_so_far = initial_size

    t_0 = time_last_display = time.time()
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        time_last_read = time.time()
        if (report_hook and
                # Refresh report every half second or when download is
                # finished.
                (time_last_read > time_last_display + 0.5 or not chunk)):
            _chunk_report_(bytes_so_far,
                           total_size, initial_size, t_0)
            time_last_display = time_last_read
        if chunk:
            local_file.write(chunk)
        else:
            break

    return


def _chunk_report_(bytes_so_far, total_size, initial_size, t_0):
    """Show downloading percentage.

    :param int bytes_so_far: number of downloaded bytes
    :param int total_size: total size of the file (may be 0/None, depending
        on download method).
    :param int t_0: the time in seconds (as returned by time.time()) at which
        the download was resumed / started.
    :param int initial_size: if resuming, indicate the initial size of the
        file. If not resuming, set to zero.
    """

    if not total_size:
        sys.stderr.write("\rDownloaded {0:d} of ? bytes.".format(bytes_so_far))

    else:
        # Estimate remaining download time
        total_percent = float(bytes_so_far) / total_size

        current_download_size = bytes_so_far - initial_size
        bytes_remaining = total_size - bytes_so_far
        delta_t = time.time() - t_0
        download_rate = current_download_size / max(1e-8, float(delta_t))
        # Minimum rate of 0.01 bytes/s, to avoid dividing by zero.
        time_remaining = bytes_remaining / max(0.01, download_rate)

        # Trailing whitespace is to erase extra char when message length
        # varies
        sys.stderr.write(
            "\rDownloaded {0:d} of {1:d} bytes ({2:.1f}%, {3!s} remaining)".format(
                bytes_so_far, total_size, total_percent * 100, _format_time(time_remaining)))


def _format_time(t_secs):
    if t_secs > 60:
        return "{0:4.1f}min".format(t_secs / 60.)
    else:
        return " {0:5.1f}s".format(t_secs)


def _md5_hash(string):
    m = hashlib.md5()
    if PY3:
        string = string.encode('utf-8')
    m.update(string)
    return m.hexdigest()
