#!/usr/bin/env python
"""Prefetch the TemplateFlow files QSIPrep uses for adult data.

The Docker build runs this script so that the image's TemplateFlow cache holds
the res-1 T1w, the res-1 T2w and the ``desc-brain`` mask of each adult template,
the files ``GetTemplate`` in ``qsiprep/interfaces/anatomical.py`` asks for.
Infant cohorts are not prefetched; QSIPrep downloads them when a run needs them.
Files already in the cache are not downloaded again.

Examples
--------
Populate the default cache (``$TEMPLATEFLOW_HOME``)::

    python scripts/fetch_templates.py

Populate a directory to bind into the container on an offline cluster::

    python scripts/fetch_templates.py --tf-dir /path/to/templateflow
"""

import argparse
import os

TEMPLATES = ('MNI152NLin2009cAsym', 'MNI152NLin6Asym')


def fetch_file(template, query):
    """Download the one file matching ``query``, or raise if there is not exactly one."""
    import templateflow.api as tf

    found = tf.get(template, **query)
    if isinstance(found, list):
        raise RuntimeError(f'{len(found)} TemplateFlow files for {template} match {query}.')
    print(found)


def fetch_template(template):
    import templateflow.api as tf

    base = {'resolution': 1, 'extension': '.nii.gz'}
    for contrast in ('T1w', 'T2w'):
        query = {**base, 'desc': None, 'suffix': contrast}
        # MNI152NLin6Asym has no T2w.
        if contrast == 'T2w' and not tf.ls(template, **query):
            print(f'{template} has no file matching {query}; skipped')
            continue
        fetch_file(template, query)
    fetch_file(template, {**base, 'desc': 'brain', 'suffix': 'mask'})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--tf-dir',
        type=os.path.abspath,
        help='TemplateFlow directory to populate (default: $TEMPLATEFLOW_HOME)',
    )
    opts = parser.parse_args()

    # TemplateFlow reads TEMPLATEFLOW_HOME when it is first imported.
    if opts.tf_dir is not None:
        os.environ['TEMPLATEFLOW_HOME'] = opts.tf_dir

    for template in TEMPLATES:
        fetch_template(template)


if __name__ == '__main__':
    main()
