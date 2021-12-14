#!/usr/bin/env python
"""
Using qsiprep within a singularity container (qsiprep-<version>.simg)


To run qsiprep with singularity, a singularity image must be built (see Installation).
This is the preferred method for HPCs. For example ::

  singularity run --cleanenv -B /data/:/home/$user/data qsiprep-<version>.simg

Please report any feedback to our GitHub repository
(https://github.com/pennbbl/qsiprep) and do not
forget to credit all the authors of software that qsiprep
uses (https://qsiprep.readthedocs.io/en/latest/citing.html).
"""
import sys
import os
import os.path as op
import re
import subprocess

__version__ = 'latest'
__packagename__ = 'qsiprep-container'
__author__ = ''
__copyright__ = 'Copyright 2019, '
__credits__ = []
__license__ = '3-clause BSD'
__maintainer__ = ''
__email__ = ''
__url__ = 'https://github.com/pennbbl/qsiprep'
__bugreports__ = 'https://github.com/pennbbl/qsiprep/issues'

__description__ = """\
qsiprep is a non-DTI diffusion-weighted image pre-processing pipeline \
that is designed to provide an easily accessible, state-of-the-art interface \
that is robust to differences in scan acquisition protocols and that requires \
minimal user input, while providing easily interpretable and comprehensive \
error and output reporting."""
__longdesc__ = """\
This package provides an intuitive interface for running the qsiprep
workflow with a singularity image."""

DOWNLOAD_URL = (
    'https://pypi.python.org/packages/source/{name[0]}/{name}/{name}-{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.7',
]


MISSING = """
Image '{}' is missing
Would you like to download? [Y/n] """
PKG_PATH = '/usr/local/miniconda/lib/python3.7/site-packages'

# Monkey-patch Py2 subprocess
if not hasattr(subprocess, 'DEVNULL'):
    subprocess.DEVNULL = -3

if not hasattr(subprocess, 'run'):
    # Reimplement minimal functionality for usage in this file
    def _run(args, stdout=None, stderr=None):
        from collections import namedtuple
        result = namedtuple('CompletedProcess', 'stdout stderr returncode')

        devnull = None
        if subprocess.DEVNULL in (stdout, stderr):
            devnull = open(os.devnull, 'r+')
            if stdout == subprocess.DEVNULL:
                stdout = devnull
            if stderr == subprocess.DEVNULL:
                stderr = devnull

        proc = subprocess.Popen(args, stdout=stdout, stderr=stderr)
        stdout, stderr = proc.communicate()
        res = result(stdout, stderr, proc.returncode)

        if devnull is not None:
            devnull.close()

        return res
    subprocess.run = _run


# De-fang Python 2's input - we don't eval user input
try:
    input = raw_input
except NameError:
    pass


def check_singularity():
    """Verify that singularity is installed and the user has permission to
    run singularity images.

    Returns
    -------
    -1  singularity can't be found
     1  Test run OK
     """
    try:
        subprocess.run(['singularity', '--version'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except OSError as e:
        from errno import ENOENT
        if e.errno == ENOENT:
            return -1
        raise e
    return 1


def check_image(image):
    """Check whether image is present and you can read it"""
    return os.path.exists(image) and os.access(image, os.R_OK)


def merge_help(wrapper_help, target_help):
    # Matches all flags with up to one nested square bracket
    opt_re = re.compile(r'(\[--?[\w-]+(?:[^\[\]]+(?:\[[^\[\]]+\])?)?\])')
    # Matches flag name only
    flag_re = re.compile(r'\[--?([\w-]+)[ \]]')

    # Normalize to Unix-style line breaks
    w_help = wrapper_help.rstrip().replace('\r', '')
    t_help = target_help.rstrip().replace('\r', '')

    w_usage, w_details = w_help.split('\n\n', 1)
    w_groups = w_details.split('\n\n')
    t_usage, t_details = t_help.split('\n\n', 1)
    t_groups = t_details.split('\n\n')

    w_posargs = w_usage.split('\n')[-1].lstrip()
    t_posargs = t_usage.split('\n')[-1].lstrip()

    w_options = opt_re.findall(w_usage)
    w_flags = sum(map(flag_re.findall, w_options), [])
    t_options = opt_re.findall(t_usage)
    t_flags = sum(map(flag_re.findall, t_options), [])

    # The following code makes this assumption
    assert w_flags[:2] == ['h', 'version']
    assert w_posargs.replace(']', '').replace('[', '') == t_posargs

    # Make sure we're not clobbering options we don't mean to
    overlap = set(w_flags).intersection(t_flags)
    expected_overlap = set(['h', 'version', 'w', 'fs-license-file'])
    assert overlap == expected_overlap, "Clobbering options: {}".format(
        ', '.join(overlap - expected_overlap))

    sections = []

    # Construct usage
    start = w_usage[:w_usage.index(' [')]
    indent = ' ' * len(start)
    new_options = sum((
        w_options[:2],
        [opt for opt, flag in zip(t_options, t_flags) if flag not in overlap],
        w_options[2:]
    ), [])
    opt_line_length = 79 - len(start)
    length = 0
    opt_lines = [start]
    for opt in new_options:
        opt = ' ' + opt
        olen = len(opt)
        if length + olen <= opt_line_length:
            opt_lines[-1] += opt
            length += olen
        else:
            opt_lines.append(indent + opt)
            length = olen
    opt_lines.append(indent + ' ' + t_posargs)
    sections.append('\n'.join(opt_lines))

    # Use target description and positional args
    sections.extend(t_groups[:2])

    for line in t_groups[2].split('\n')[1:]:
        content = line.lstrip().split(',', 1)[0]
        if content[1:] not in overlap:
            w_groups[2] += '\n' + line

    sections.append(w_groups[2])

    # All remaining sections, show target then wrapper (skipping duplicates)
    sections.extend(t_groups[3:] + w_groups[6:])
    return '\n\n'.join(sections)


def get_parser():
    """Defines the command line interface of the wrapper"""
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    # Standard qsiprep arguments
    parser.add_argument('bids_dir', nargs='?', type=os.path.abspath,
                        default='')
    parser.add_argument('output_dir', nargs='?', type=os.path.abspath,
                        default='')
    parser.add_argument('analysis_level', nargs='?', choices=['participant'],
                        default='participant')

    # For qsirecon
    parser.add_argument('--recon-input', '--recon_input',
                        required=False,
                        action='store',
                        type=os.path.abspath)
    parser.add_argument('--bids-filter-file', '--bids_filter_file',
                        required=False,
                        action='store',
                        type=os.path.abspath)

    parser.add_argument('-h', '--help', action='store_true',
                        help="show this help message and exit")
    parser.add_argument('--version', action='store_true',
                        help="show program's version number and exit")

    # Allow alternative images (semi-developer)

    parser.add_argument('-i', '--image', metavar='IMG', type=str,
                        default=os.path.expanduser("~") + "/qsiprep-latest.simg",
                        help='image name')

    # Options for mapping files and directories into container
    # Update `expected_overlap` variable in merge_help() when adding to this
    g_wrap = parser.add_argument_group(
        'Wrapper options',
        'Standard options that require mapping files into the container')
    g_wrap.add_argument('-w', '--work-dir', action='store', type=os.path.abspath,
                        help='path where intermediate results should be stored')
    g_wrap.add_argument(
        '--fs-license-file', metavar='PATH', type=os.path.abspath,
        default=os.getenv('FS_LICENSE', None),
        help='Path to FreeSurfer license key file. Get it (for free) by registering'
             ' at https://surfer.nmr.mgh.harvard.edu/registration.html')
    g_wrap.add_argument('--recon-spec', '--recon_spec',
                        required=False,
                        action='store',
                        type=str)
    g_wrap.add_argument('--eddy-config', '--eddy_config',
                        required=False,
                        action='store',
                        type=os.path.abspath)

    # Developer patch/shell options
    g_dev = parser.add_argument_group(
        'Developer options',
        'Tools for testing and debugging qsiprep')
    g_dev.add_argument('-f', '--patch-qsiprep', metavar='PATH',
                       type=os.path.abspath,
                       help='working qsiprep repository')
    g_dev.add_argument('-p', '--patch-nipype', metavar='PATH',
                       type=os.path.abspath,
                       help='working nipype repository')
    g_dev.add_argument('-a', '--custom_atlases',
                       type=os.path.abspath,
                       help='custom atlas directory')
    g_dev.add_argument('--shell', action='store_true',
                       help='open shell in image instead of running qsiprep')
    g_dev.add_argument('--config', metavar='PATH', action='store',
                       type=os.path.abspath, help='Use custom nipype.cfg file')
    g_dev.add_argument('-e', '--env', action='append', nargs=2, metavar=('ENV_VAR', 'value'),
                       help='Set custom environment variable within container')
    g_dev.add_argument('-u', '--user', action='store',
                       help='Run container as a given user/uid')

    return parser


def mkdir(dirpath):
    if op.exists(dirpath):
        return 1
    try:
        os.makedirs(dirpath)
    except Exception:
        print("Unable to create {}. Exiting.".format(dirpath))
        sys.exit(1)


def main():
    """Entry point"""

    parser = get_parser()
    print(sys.argv)
    # Capture additional arguments to pass inside container
    opts, unknown_args = parser.parse_known_args()

    # Set help if no directories set
    if (opts.bids_dir, opts.output_dir, opts.version) == ('', '', False):
        opts.help = True

    # Stop if no docker / docker fails to run
    check = check_singularity()
    if check < 1:
        if opts.version:
            print('qsiprep wrapper {!s}'.format(__version__))
        if opts.help:
            parser.print_help()
        else:
            print("qsiprep: Could not find singularity command... Is it installed?")
        return 1

    # For --help or --version, ask before downloading an image
    if not check_image(opts.image):
        resp = 'Y'
        if opts.version:
            print('qsiprep wrapper {!s}'.format(__version__))
        if opts.help:
            parser.print_help()
        if opts.version or opts.help:
            try:
                resp = input(MISSING.format(opts.image))
            except KeyboardInterrupt:
                print()
                return 1
        if resp not in ('y', 'Y', ''):
            return 0
        print('Downloading and building image. This may take a while...')
        ret = subprocess.run(
            "singularity build {} docker://pennbbl/qsiprep:latest".format(opts.image))
        if ret > 0:
            print("Critical Error: Unable to create singularity image {}".format(opts.image))
            sys.exit(1)

    # Warn on low memory allocation
    # Warn on low memory allocation
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes/(1024.**3)
    if mem_gib < 10:
        print('Warning: <10GB of RAM is available on your system.\n'
              'Some parts of QSIPrep may fail to complete.')
    if not (opts.help or opts.version or '--reports-only' in unknown_args) and mem_gib < 10:
        print('Warning: <10GB of RAM is available within your '
              'environment.\nSome parts of qsiprep may fail to complete.')
        if '--mem_mb' not in unknown_args:
            resp = 'N'
            try:
                resp = input('Continue anyway? [y/N]')
            except KeyboardInterrupt:
                print()
                return 1
            if resp not in ('y', 'Y', ''):
                return 0

    command = ['singularity', 'run', '--cleanenv']

    # Patch working repositories into installed package directories
    for pkg in ('qsiprep', 'nipype'):
        repo_path = getattr(opts, 'patch_' + pkg)
        if repo_path is not None:
            command.extend(['-B',
                            '{}:{}/{}'.format(repo_path, PKG_PATH, pkg)])

    main_args = ['/sngl/data', '/sngl/out', opts.analysis_level]
    if opts.fs_license_file:
        license_dir, license_fname = op.split(opts.fs_license_file)
        mounted_license = "/mnt/" + license_fname
        command.extend(['-B', license_dir + ':/mnt'])
        main_args.extend(['--fs-license-file', mounted_license])
    if opts.bids_dir:
        command.extend(['-B', ':'.join((opts.bids_dir, '/sngl/data'))])
    if opts.recon_input:
        command.extend(['-B', ':'.join((opts.recon_input, '/sngl/qsiprep-output'))])
        main_args.extend(['--recon-input', '/sngl/qsiprep-output'])
    if opts.recon_spec:
        if os.path.exists(opts.recon_spec):
            spec_dir, spec_fname = op.split(opts.recon_spec)
            mounted_spec = "/sngl/spec/" + spec_fname
            command.extend(['-B', ':'.join((spec_dir, '/sngl/spec'))])
            main_args.extend(['--recon-spec', mounted_spec])
        else:
            main_args.extend(['--recon-spec', opts.recon_spec])
    if opts.eddy_config:
        config_dir, config_fname = op.split(opts.eddy_config)
        mounted_config = "/sngl/eddy/" + config_fname
        command.extend(['-B', ':'.join((config_dir, '/sngl/eddy'))])
        main_args.extend(['--eddy-config', mounted_config])
    if opts.bids_filter_file:
        filter_dir, filter_fname = op.split(opts.bids_filter_file)
        mounted_filter = "/sngl/filter/" + filter_fname
        command.extend(['-B', ':'.join((filter_dir, '/sngl/filter'))])
        main_args.extend(['--bids-filter-file', mounted_filter])
    if opts.output_dir:
        mkdir(opts.output_dir)
        command.extend(['-B', ':'.join((opts.output_dir, '/sngl/out'))])
    if opts.custom_atlases:
        command.extend(['-B', ':'.join((opts.custom_atlases, '/atlas/qsirecon_atlases'))])
    if opts.work_dir:
        mkdir(opts.work_dir)
        command.extend(['-B', ':'.join((opts.work_dir, '/sngl/scratch'))])
        unknown_args.extend(['-w', '/sngl/scratch'])

    if opts.config:
        config_dir, config_fname = op.split(opts.config)
        command.extend(['-B', ':'.join((config_dir,
                                        '/root/.nipype'))])

    if opts.shell:
        command[1] = 'shell'

    command.append(opts.image)

    # Override help and version to describe underlying program
    # Respects '-i' flag, so will retrieve information from any image
    if opts.help:
        command.append('-h')
        targethelp = subprocess.check_output(command).decode()
        print(merge_help(parser.format_help(), targethelp))
        return 0
    elif opts.version:
        # Get version to be run and exit
        command.append('--version')
        ret = subprocess.run(command)
        return ret.returncode

    if not opts.shell:
        command.extend(main_args)
        command.extend(unknown_args)

    print("RUNNING: " + ' '.join(command))
    ret = subprocess.run(command)
    if ret.returncode:
        print("qsiprep: Please report errors to {}".format(__bugreports__))
    return ret.returncode


if __name__ == '__main__':
    sys.exit(main())
