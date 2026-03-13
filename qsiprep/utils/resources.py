# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Adapt importlib.resources result to a filesystem path string.

files() may return a Traversable (with .as_file()) or a pathlib.Path when the
package is on the filesystem; this helper supports both.
"""


def as_path(resource):
    """Return a filesystem path string from the result of files() / resource."""
    if hasattr(resource, 'as_file'):
        with resource.as_file() as path:
            return str(path)
    return str(resource)
