#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
"""


__all__ = ['fix_multi_T1w_source_name', 'add_suffix', 'read_crashfile',
           'splitext', '_copy_any']


def fix_multi_T1w_source_name(in_files):
    """
    Make up a generic source name when there are multiple T1s

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'

    """
    import os
    from nipype.utils.filemanip import filename_to_list
    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)


def add_suffix(in_files, suffix):
    """
    Wrap nipype's fname_presuffix to conveniently just add a prefix

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'

    """
    import os.path as op
    from nipype.utils.filemanip import fname_presuffix, filename_to_list
    return op.basename(fname_presuffix(filename_to_list(in_files)[0],
                                       suffix=suffix))


def read_crashfile(path):
    if path.endswith('.pklz'):
        return _read_pkl(path)
    elif path.endswith('.txt'):
        return _read_txt(path)
    raise RuntimeError('unknown crashfile format')


def _read_pkl(path):
    from nipype.utils.filemanip import loadcrash
    crash_data = loadcrash(path)
    data = {'file': path,
            'traceback': ''.join(crash_data['traceback'])}
    if 'node' in crash_data:
        data['node'] = crash_data['node']
        if data['node'].base_dir:
            data['node_dir'] = data['node'].output_dir()
        else:
            data['node_dir'] = "Node crashed before execution"
        data['inputs'] = sorted(data['node'].inputs.trait_get().items())
    return data


def _read_txt(path):
    """Read a txt crashfile

    >>> new_path = Path(__file__).resolve().parent.parent
    >>> test_data_path = new_path / 'data' / 'tests'
    >>> info = _read_txt(test_data_path / 'crashfile.txt')
    >>> info['node']  # doctest: +ELLIPSIS
    '...func_preproc_task_machinegame_run_02_wf.carpetplot_wf.conf_plot'
    >>> info['traceback']  # doctest: +ELLIPSIS
    '...ValueError: zero-size array to reduction operation minimum which has no identity'

    """
    from pathlib import Path
    lines = Path(path).read_text().splitlines()
    data = {'file': str(path)}
    traceback_start = 0
    if lines[0].startswith('Node'):
        data['node'] = lines[0].split(': ', 1)[1].strip()
        data['node_dir'] = lines[1].split(': ', 1)[1].strip()
        inputs = []
        cur_key = ''
        cur_val = ''
        for i, line in enumerate(lines[5:]):
            if not line.strip():
                continue

            if line[0].isspace():
                cur_val += line
                continue

            if cur_val:
                inputs.append((cur_key, cur_val.strip()))

            if line.startswith("Traceback ("):
                traceback_start = i + 5
                break

            cur_key, cur_val = tuple(line.split(' = ', 1))

        data['inputs'] = sorted(inputs)
    else:
        data['node_dir'] = "Node crashed before execution"
    data['traceback'] = '\n'.join(lines[traceback_start:]).strip()
    return data


def splitext(fname):
    """Splits filename and extension (.gz safe)

    >>> splitext('some/file.nii.gz')
    ('file', '.nii.gz')
    >>> splitext('some/other/file.nii')
    ('file', '.nii')
    >>> splitext('otherext.tar.gz')
    ('otherext', '.tar.gz')
    >>> splitext('text.txt')
    ('text', '.txt')

    """
    from pathlib import Path
    basename = str(Path(fname).name)
    stem = Path(basename.rstrip('.gz')).stem
    return stem, basename[len(stem):]


def _copy_any(src, dst):
    import os
    import gzip
    from shutil import copyfileobj
    from nipype.utils.filemanip import copyfile

    src_isgz = src.endswith('.gz')
    dst_isgz = dst.endswith('.gz')
    if not src_isgz and not dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    with src_open(src, 'rb') as f_in:
        with open(dst, 'wb') as f_out:
            if dst_isgz:
                # Remove FNAME header from gzip (poldracklab/fmriprep#1480)
                gz_out = gzip.GzipFile('', 'wb', 9, f_out, 0.)
                copyfileobj(f_in, gz_out)
                gz_out.close()
            else:
                copyfileobj(f_in, f_out)

    return True


if __name__ == '__main__':
    pass
