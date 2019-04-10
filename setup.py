#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
""" qsiprep setup script """


def main():
    """ Install entry-point """
    from io import open
    from os import path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from numpy import get_include
    from qsiprep.__about__ import (
        __packagename__,
        __version__,
        __author__,
        __email__,
        __maintainer__,
        __license__,
        __description__,
        __longdesc__,
        __url__,
        DOWNLOAD_URL,
        CLASSIFIERS,
        REQUIRES,
        SETUP_REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
    )

    pkg_data = {
        'qsiprep': [
            'data/*.json',
            'data/*.nii.gz',
            'data/*.mat',
            'data/boilerplate.bib',
            'data/itkIdentityTransform.txt',
            'viz/*.tpl',
            'viz/*.json',
            'niworkflows/data/t1w-mni_registration*.json',
            'niworkflows/data/bold-mni_registration*.json',
        ]
    }

    root_dir = op.dirname(op.abspath(getfile(currentframe())))

    version = None
    cmdclass = {}
    if op.isfile(op.join(root_dir, 'qsiprep', 'VERSION')):
        with open(op.join(root_dir, 'qsiprep', 'VERSION')) as vfile:
            version = vfile.readline().strip()
        pkg_data['qsiprep'].insert(0, 'VERSION')

    if version is None:
        import versioneer
        version = versioneer.get_version()
        cmdclass = versioneer.get_cmdclass()

    extensions = [Extension(
        "qsiprep.utils.maths",
        ["qsiprep/utils/maths.pyx"],
        include_dirs=[get_include(), "/usr/local/include/"],
        library_dirs=["/usr/lib/"]),
    ]

    setup(
        name=__packagename__,
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        license=__license__,
        classifiers=CLASSIFIERS,
        download_url=DOWNLOAD_URL,
        # Dependencies handling
        setup_requires=SETUP_REQUIRES,
        install_requires=REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        dependency_links=LINKS_REQUIRES,
        package_data=pkg_data,
        entry_points={'console_scripts': [
            'qsiprep=qsiprep.cli.run:main',
            'mif2fib=qsiprep.cli.convertODFs:mif_to_fib',
            'fib2mif=qsiprep.cli.convertODFs:fib_to_mif'
        ]},
        packages=find_packages(exclude=("tests",)),
        zip_safe=False,
        ext_modules=extensions,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    main()
