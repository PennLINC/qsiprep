# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Utilities for tracking and filtering spaces.

Vendored from niworkflows and modified for QSIPrep's purposes.
"""

import argparse
from collections import defaultdict
from itertools import product

import attr
from templateflow import api as _tfapi

NONSTANDARD_REFERENCES = [
    'ACPC',  # QSIPrep's primary output space
    'T1w',
    'T2w',
    'anat',
    'fsnative',
    'func',
    'run',
    'sbref',
    'session',
    'individual',
    'dwi',
    'asl',
]
"""List of supported nonstandard reference spaces."""

NONSTANDARD_2D_REFERENCES = ['fsnative']
"""List of supported nonstandard 2D reference spaces."""

FSAVERAGE_DENSITY = {
    'fsaverage3': '642',
    'fsaverage4': '2562',
    'fsaverage5': '10k',
    'fsaverage6': '41k',
    'fsaverage': '164k',
}
"""A map of legacy fsaverageX names to surface densities."""

FSAVERAGE_LEGACY = {v: k for k, v in FSAVERAGE_DENSITY.items()}
"""A map of surface densities to legacy fsaverageX names."""


@attr.s(slots=True, frozen=True)
class Reference:
    """
    Represent a (non)standard space specification.

    Examples
    --------
    >>> Reference('MNI152NLin2009cAsym')
    Reference(space='MNI152NLin2009cAsym', spec={})

    >>> Reference('MNI152NLin2009cAsym', {})
    Reference(space='MNI152NLin2009cAsym', spec={})

    >>> Reference('MNI152NLin2009cAsym', None)
    Reference(space='MNI152NLin2009cAsym', spec={})

    >>> Reference('MNI152NLin2009cAsym', {'res': 1})
    Reference(space='MNI152NLin2009cAsym', spec={'res': 1})

    >>> Reference('MNIPediatricAsym', {'cohort': '1'})
    Reference(space='MNIPediatricAsym', spec={'cohort': '1'})

    >>> Reference('func')
    Reference(space='func', spec={})

    >>> # Checks spaces with cohorts:
    >>> Reference('MNIPediatricAsym')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNIPediatricAsym" is not fully defined.
    ...

    >>> Reference(space='MNI152Lin', spec={'cohort': 1})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNI152Lin" does not accept ...

    >>> Reference('MNIPediatricAsym', {'cohort': '100'})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: standard space "MNIPediatricAsym" does not contain ...

    >>> Reference('MNIPediatricAsym', 'blah')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: ...

    >>> Reference('shouldraise')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: space identifier "shouldraise" is invalid.
    ...

    >>> # Check standard property
    >>> Reference('func').standard
    False
    >>> Reference('MNI152Lin').standard
    True
    >>> Reference('MNIPediatricAsym', {'cohort': 1}).standard
    True

    >>> # Check dim property
    >>> Reference('func').dim
    3
    >>> Reference('MNI152NLin6Asym').dim
    3
    >>> Reference('fsnative').dim
    2
    >>> Reference('onavg').dim
    2

    >>> # Equality/inequality checks
    >>> Reference('func') == Reference('func')
    True
    >>> Reference('func') != Reference('MNI152Lin')
    True
    >>> Reference('MNI152Lin', {'res': 1}) == Reference('MNI152Lin', {'res': 1})
    True
    >>> Reference('MNI152Lin', {'res': 1}) == Reference('MNI152Lin', {'res': 2})
    False
    >>> sp1 = Reference('MNIPediatricAsym', {'cohort': 1})
    >>> sp2 = Reference('MNIPediatricAsym', {'cohort': 2})
    >>> sp1 == sp2
    False
    >>> sp1 = Reference('MNIPediatricAsym', {'res': 1, 'cohort': 1})
    >>> sp2 = Reference('MNIPediatricAsym', {'cohort': 1, 'res': 1})
    >>> sp1 == sp2
    True

    """

    _standard_spaces = tuple(_tfapi.templates())
    _spaces_2d = tuple(_tfapi.templates(suffix='sphere'))

    space = attr.ib(default=None, type=str)
    """Name designating this space."""
    spec = attr.ib(
        factory=dict,
        validator=attr.validators.optional(attr.validators.instance_of(dict)),
    )
    """The dictionary of specs."""
    standard = attr.ib(default=False, repr=False, type=bool)
    """Whether this space is standard or not."""
    dim = attr.ib(default=3, repr=False, type=int)
    """Dimensionality of the sampling manifold."""

    def __attrs_post_init__(self):
        """Extract cohort out of spec."""
        if self.spec is None:
            object.__setattr__(self, 'spec', {})

        if self.space.startswith('fsaverage'):
            space = self.space
            object.__setattr__(self, 'space', 'fsaverage')

            if 'den' not in self.spec or space != 'fsaverage':
                spec = self.spec.copy()
                spec['den'] = FSAVERAGE_DENSITY[space]
                object.__setattr__(self, 'spec', spec)

        if (self.space in self._spaces_2d) or (self.space in NONSTANDARD_2D_REFERENCES):
            object.__setattr__(self, 'dim', 2)

        if self.space in self._standard_spaces:
            object.__setattr__(self, 'standard', True)

        _cohorts = [f'{t}' for t in _tfapi.TF_LAYOUT.get_cohorts(template=self.space)]
        if 'cohort' in self.spec:
            if not _cohorts:
                raise ValueError(
                    f'standard space "{self.space}" does not accept a cohort specification.'
                )

            if str(self.spec['cohort']) not in _cohorts:
                raise ValueError(
                    'standard space "{}" does not contain any cohort '
                    'named "{}".'.format(self.space, self.spec['cohort'])
                )
        elif _cohorts:
            _cohorts = ', '.join([f'"cohort-{c}"' for c in _cohorts])
            print(
                f'Standard space "{self.space}" is not fully defined.\n'
                'QSIPrep will attempt to infer the appropriate cohort '
                f'from the participant age from {_cohorts}.'
            )

    @property
    def fullname(self):
        """
        Generate a full-name combining cohort.

        Examples
        --------
        >>> Reference('MNI152Lin').fullname
        'MNI152Lin'

        >>> Reference('MNIPediatricAsym', {'cohort': 1}).fullname
        'MNIPediatricAsym:cohort-1'

        """
        if 'cohort' not in self.spec:
            return self.space
        return '{}:cohort-{}'.format(self.space, self.spec['cohort'])

    @property
    def legacyname(self):
        """
        Generate a legacy name for fsaverageX spaces.

        Examples
        --------
        >>> Reference(space='fsaverage')
        Reference(space='fsaverage', spec={'den': '164k'})
        >>> Reference(space='fsaverage').legacyname
        'fsaverage'
        >>> Reference(space='fsaverage6')
        Reference(space='fsaverage', spec={'den': '41k'})
        >>> Reference(space='fsaverage6').legacyname
        'fsaverage6'
        >>> # Overwrites density of legacy "fsaverage" specifications
        >>> Reference(space='fsaverage6', spec={'den': '10k'})
        Reference(space='fsaverage', spec={'den': '41k'})
        >>> Reference(space='fsaverage6', spec={'den': '10k'}).legacyname
        'fsaverage6'
        >>> # Return None if no legacy space
        >>> Reference(space='fsaverage', spec={'den': '30k'}).legacyname is None
        True

        """
        if self.space == 'fsaverage' and self.spec['den'] in FSAVERAGE_LEGACY:
            return FSAVERAGE_LEGACY[self.spec['den']]

    @space.validator
    def _check_name(self, attribute, value):
        if value.startswith('fsaverage'):
            return
        valid = list(self._standard_spaces) + NONSTANDARD_REFERENCES
        if value not in valid:
            raise ValueError(
                'space identifier "{}" is invalid.\nValid '
                'identifiers are: {}'.format(value, ', '.join(valid))
            )

    def __str__(self):
        """
        Format this reference.

        Examples
        --------
        >>> str(Reference(space='MNIPediatricAsym', spec={'cohort': 2, 'res': 1}))
        'MNIPediatricAsym:cohort-2:res-1'

        """
        return ':'.join(
            [self.space] + ['-'.join((k, str(v))) for k, v in sorted(self.spec.items())]
        )

    @classmethod
    def from_string(cls, value):
        """
        Parse a string to generate the corresponding list of References.

        .. testsetup::

            >>> if PY_VERSION < (3, 6):
            ...     pytest.skip("This doctest does not work on python <3.6")

        Parameters
        ----------
        value: :obj:`str`
            A string containing a space specification following *fMRIPrep*'s
            language for ``--output-spaces``
            (e.g., ``MNIPediatricAsym:cohort-1:cohort-2:res-1:res-2``).

        Returns
        -------
        spaces : :obj:`list` of :obj:`Reference`
            A list of corresponding spaces given the input string.

        Examples
        --------
        >>> Reference.from_string("MNI152NLin2009cAsym")
        [Reference(space='MNI152NLin2009cAsym', spec={})]

        >>> # Bad space name
        >>> Reference.from_string("shouldraise")
        Traceback (most recent call last):
          ...
        ValueError: space identifier "shouldraise" is invalid.
        ...

        >>> # Missing cohort
        >>> Reference.from_string("MNIPediatricAsym")
        Traceback (most recent call last):
          ...
        ValueError: standard space "MNIPediatricAsym" is not fully defined.
        ...

        >>> Reference.from_string("MNIPediatricAsym:cohort-1")
        [Reference(space='MNIPediatricAsym', spec={'cohort': '1'})]

        >>> Reference.from_string(
        ...     "MNIPediatricAsym:cohort-1:cohort-2"
        ... )  # doctest: +NORMALIZE_WHITESPACE
        [Reference(space='MNIPediatricAsym', spec={'cohort': '1'}),
         Reference(space='MNIPediatricAsym', spec={'cohort': '2'})]

        >>> Reference.from_string("fsaverage:den-10k:den-164k")  # doctest: +NORMALIZE_WHITESPACE
        [Reference(space='fsaverage', spec={'den': '10k'}),
         Reference(space='fsaverage', spec={'den': '164k'})]

        >>> Reference.from_string(
        ...     "MNIPediatricAsym:cohort-5:cohort-6:res-2"
        ... )  # doctest: +NORMALIZE_WHITESPACE
        [Reference(space='MNIPediatricAsym', spec={'cohort': '5', 'res': '2'}),
         Reference(space='MNIPediatricAsym', spec={'cohort': '6', 'res': '2'})]

        >>> Reference.from_string(
        ...     "MNIPediatricAsym:cohort-5:cohort-6:res-2:res-iso1.6mm"
        ... )  # doctest: +NORMALIZE_WHITESPACE
        [Reference(space='MNIPediatricAsym', spec={'cohort': '5', 'res': '2'}),
         Reference(space='MNIPediatricAsym', spec={'cohort': '5', 'res': 'iso1.6mm'}),
         Reference(space='MNIPediatricAsym', spec={'cohort': '6', 'res': '2'}),
         Reference(space='MNIPediatricAsym', spec={'cohort': '6', 'res': 'iso1.6mm'})]

        """
        _args = value.split(':')
        spec = defaultdict(list, {})
        for modifier in _args[1:]:
            mitems = modifier.split('-', 1)
            spec[mitems[0]].append(len(mitems) == 1 or mitems[1])

        allspecs = _expand_entities(spec)

        return [cls(_args[0], s) for s in allspecs]


class SpatialReferences:
    """
    Manage specifications of spatial references.

    Examples
    --------
    >>> sp = SpatialReferences([
    ...     'func',
    ...     'fsnative',
    ...     'MNI152NLin2009cAsym',
    ...     'anat',
    ...     'fsaverage5',
    ...     'fsaverage6',
    ...     ('MNIPediatricAsym', {'cohort': '2'}),
    ...     ('MNI152NLin2009cAsym', {'res': 2}),
    ...     ('MNI152NLin2009cAsym', {'res': 1}),
    ... ])
    >>> sp.get_spaces(standard=False)
    ['func', 'fsnative', 'anat']

    >>> sp.get_spaces(standard=False, dim=(3,))
    ['func', 'anat']

    >>> sp.get_spaces(nonstandard=False)
    ['MNI152NLin2009cAsym', 'fsaverage', 'MNIPediatricAsym:cohort-2']

    >>> sp.get_spaces(nonstandard=False, dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2']

    >>> sp.get_fs_spaces()
    ['fsnative', 'fsaverage5', 'fsaverage6']

    >>> sp.get_standard(full_spec=True)  # doctest: +NORMALIZE_WHITESPACE
    [Reference(space='fsaverage', spec={'den': '10k'}),
     Reference(space='fsaverage', spec={'den': '41k'}),
     Reference(space='MNI152NLin2009cAsym', spec={'res': 2}),
     Reference(space='MNI152NLin2009cAsym', spec={'res': 1})]

    >>> sp.is_cached()
    False
    >>> sp.cached  # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: References have not ...

    >>> sp.checkpoint()
    >>> sp.is_cached()
    True
    >>> sp.cached.references  # doctest: +NORMALIZE_WHITESPACE
    [Reference(space='func', spec={}),
     Reference(space='fsnative', spec={}),
     Reference(space='MNI152NLin2009cAsym', spec={}),
     Reference(space='anat', spec={}),
     Reference(space='fsaverage', spec={'den': '10k'}),
     Reference(space='fsaverage', spec={'den': '41k'}),
     Reference(space='MNIPediatricAsym', spec={'cohort': '2'}),
     Reference(space='MNI152NLin2009cAsym', spec={'res': 2}),
     Reference(space='MNI152NLin2009cAsym', spec={'res': 1})]

    >>> sp.cached.get_fs_spaces()
    ['fsnative', 'fsaverage5', 'fsaverage6']

    >>> sp.add(('MNIPediatricAsym', {'cohort': '2'}))
    >>> sp.get_spaces(nonstandard=False, dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2']

    >>> sp += [('MNIPediatricAsym', {'cohort': '2'})]  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: space ...

    >>> sp += [('MNIPediatricAsym', {'cohort': '1'})]
    >>> sp.get_spaces(nonstandard=False, dim=(3,))
    ['MNI152NLin2009cAsym', 'MNIPediatricAsym:cohort-2', 'MNIPediatricAsym:cohort-1']

    >>> sp.insert(0, ('MNIPediatricAsym', {'cohort': '3'}))
    >>> sp.get_spaces(nonstandard=False, dim=(3,))  # doctest: +NORMALIZE_WHITESPACE
    ['MNIPediatricAsym:cohort-3',
     'MNI152NLin2009cAsym',
     'MNIPediatricAsym:cohort-2',
     'MNIPediatricAsym:cohort-1']

    >>> sp.insert(0, ('MNIPediatricAsym', {'cohort': '3'}))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: space ...

    >>> sp.checkpoint()  # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: References have already ...

    >>> sp.checkpoint(force=True)
    >>> sp.cached.references  # doctest: +NORMALIZE_WHITESPACE
    [Reference(space='MNIPediatricAsym', spec={'cohort': '3'}),
     Reference(space='func', spec={}),
     Reference(space='fsnative', spec={}),
     Reference(space='MNI152NLin2009cAsym', spec={}),
     Reference(space='anat', spec={}),
     Reference(space='fsaverage', spec={'den': '10k'}),
     Reference(space='fsaverage', spec={'den': '41k'}),
     Reference(space='MNIPediatricAsym', spec={'cohort': '2'}),
     Reference(space='MNI152NLin2009cAsym', spec={'res': 2}),
     Reference(space='MNI152NLin2009cAsym', spec={'res': 1}),
     Reference(space='MNIPediatricAsym', spec={'cohort': '1'})]

    """

    __slots__ = ('_refs', '_cached')
    standard_spaces = tuple(_tfapi.templates())
    """List of supported standard reference spaces."""

    @staticmethod
    def check_space(space):
        """Build a :class:`Reference` object."""
        try:
            if isinstance(space, Reference):
                return space
        except IndexError:
            pass

        spec = {}
        if not isinstance(space, str):
            try:
                spec = space[1] or {}
            except IndexError:
                pass
            except TypeError:
                space = (None,)

            space = space[0]
        return Reference(space, spec)

    def __init__(self, spaces=None, checkpoint=False):
        """
        Maintain the bookkeeping of spaces and templates.

        Internal spaces are normalizations required for pipeline execution which
        can vary based on user arguments.
        Output spaces are desired user outputs.
        """
        self._refs = []
        self._cached = None
        if spaces is not None:
            if isinstance(spaces, str):
                spaces = [spaces]
            self.__iadd__(spaces)

            if checkpoint is True:
                self.checkpoint()

    def __iadd__(self, b):
        """Append a list of transforms to the internal list."""
        if not isinstance(b, list | tuple):
            raise TypeError('Must be a list.')

        for space in b:
            self.append(space)
        return self

    def __contains__(self, item):
        """Implement the ``in`` builtin."""
        if not self.references:
            return False
        item = self.check_space(item)
        for s in self.references:
            if s == item:
                return True
        return False

    def __str__(self):
        """
        Representation of this object.

        Examples
        --------
        >>> print(SpatialReferences())
        Spatial References: <none>.

        >>> print(SpatialReferences(['MNI152NLin2009cAsym']))
        Spatial References: MNI152NLin2009cAsym

        >>> print(SpatialReferences(['MNI152NLin2009cAsym', 'fsaverage5']))
        Spatial References: MNI152NLin2009cAsym, fsaverage:den-10k

        """
        spaces = ', '.join([str(s) for s in self.references]) or '<none>.'
        return f'Spatial References: {spaces}'

    @property
    def references(self):
        """Get all specified references."""
        return self._refs

    @property
    def cached(self):
        """Get cached spaces, raise error if not cached."""
        if not self.is_cached():
            raise ValueError('References have not been cached')
        return self._cached

    def is_cached(self):
        return self._cached is not None

    def checkpoint(self, force=False):
        """Cache and freeze current spaces to separate attribute."""
        if self.is_cached() and not force:
            raise ValueError('References have already been cached')
        self._cached = self.__class__(self.references)

    def add(self, value):
        """Add one more space, without erroring if it exists."""
        if value not in self:
            self._refs += [self.check_space(value)]

    def append(self, value):
        """Concatenate one more space."""
        if value not in self:
            self._refs += [self.check_space(value)]
            return

        raise ValueError(f'space "{str(value)}" already in spaces.')

    def insert(self, index, value, error=True):
        """Concatenate one more space."""
        if value not in self:
            self._refs.insert(index, self.check_space(value))
        elif error is True:
            raise ValueError(f'space "{str(value)}" already in spaces.')

    def get_spaces(self, standard=True, nonstandard=True, dim=(2, 3)):
        """
        Return space names.

        Parameters
        ----------
        standard : :obj:`bool`, optional
            Return standard spaces.
        nonstandard : :obj:`bool`, optional
            Return nonstandard spaces.
        dim : :obj:`tuple`, optional
            Desired dimensions of the standard spaces (default is ``(2, 3)``)

        Examples
        --------
        >>> spaces = SpatialReferences(['MNI152NLin6Asym', ("fsaverage", {"den": "10k"})])
        >>> spaces.get_spaces()
        ['MNI152NLin6Asym', 'fsaverage']

        >>> spaces.get_spaces(standard=False)
        []

        >>> spaces.get_spaces(dim=(3,))
        ['MNI152NLin6Asym']

        >>> spaces.add(('MNI152NLin6Asym', {'res': '2'}))
        >>> spaces.get_spaces()
        ['MNI152NLin6Asym', 'fsaverage']

        >>> spaces.add(('func', {}))
        >>> spaces.get_spaces()
        ['MNI152NLin6Asym', 'fsaverage', 'func']

        >>> spaces.get_spaces(nonstandard=False)
        ['MNI152NLin6Asym', 'fsaverage']

        >>> spaces.get_spaces(standard=False)
        ['func']

        """
        out = []
        for s in self.references:
            if (
                s.fullname not in out
                and (s.standard is standard or s.standard is not nonstandard)
                and s.dim in dim
            ):
                out.append(s.fullname)
        return out

    def get_standard(self, full_spec=False, dim=(2, 3)):
        """
        Return output spaces.

        Parameters
        ----------
        full_spec : :obj:`bool`
            Return only fully-specified standard references (i.e., they must either
            have density or resolution set).
        dim : :obj:`tuple`, optional
            Desired dimensions of the standard spaces (default is ``(2, 3)``)

        """
        if not full_spec:
            return [s for s in self.references if s.standard and s.dim in dim]

        return [
            s
            for s in self.references
            if s.standard and s.dim in dim and (hasspec('res', s.spec) or hasspec('den', s.spec))
        ]

    def get_nonstandard(self, full_spec=False, dim=(2, 3)):
        """Return nonstandard spaces."""
        if not full_spec:
            return [s.space for s in self.references if not s.standard and s.dim in dim]
        return [
            s.space
            for s in self.references
            if not s.standard
            and s.dim in dim
            and (hasspec('res', s.spec) or hasspec('den', s.spec))
        ]

    def get_fs_spaces(self):
        """
        Return FreeSurfer spaces.

        Discards nonlegacy fsaverage values (i.e., with nonstandard density value).

        Examples
        --------
        >>> SpatialReferences([
        ...     'fsnative',
        ...     'fsaverage6',
        ...     'fsaverage5',
        ...     'MNI152NLin6Asym',
        ... ]).get_fs_spaces()
        ['fsnative', 'fsaverage6', 'fsaverage5']

        >>> SpatialReferences([
        ...     'fsnative',
        ...     'fsaverage6',
        ...     Reference(space='fsaverage', spec={'den': '30k'})
        ... ]).get_fs_spaces()
        ['fsnative', 'fsaverage6']

        """
        return [
            s.legacyname or s.space
            for s in self.references
            if s.legacyname or s.space == 'fsnative'
        ]


class OutputReferencesAction(argparse.Action):
    """Parse spatial references."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Execute parser."""
        spaces = getattr(namespace, self.dest) or SpatialReferences()
        if not values:
            # option was called without any output spaces, so user does not want outputs
            spaces.checkpoint()
        for val in values:
            val = val.rstrip(':')
            if (
                val not in NONSTANDARD_REFERENCES
                and not val.split(':')[0].startswith('fs')
                and ':res-' not in val
                and ':resolution-' not in val
            ):
                # by default, explicitly set volumetric resolution to native
                # relevant discussions:
                # https://github.com/nipreps/niworkflows/pull/457#discussion_r375510227
                # https://github.com/nipreps/niworkflows/pull/494
                val = ':'.join((val, 'res-native'))
            for sp in Reference.from_string(val):
                spaces.add(sp)
        setattr(namespace, self.dest, spaces)


def hasspec(value, specs):
    """Check whether any of the keys are in a dict."""
    for s in specs:
        if s in value:
            return True
    return False


def format_reference(in_tuple):
    """
    Format a spatial reference given as a tuple.

    Examples
    --------
    >>> format_reference(('MNI152Lin', {'res': 1}))
    'MNI152Lin_res-1'
    >>> format_reference(('MNIPediatricAsym:cohort-2', {'res': 2}))
    'MNIPediatricAsym_cohort-2_res-2'

    """
    out = in_tuple[0].split(':')
    res = in_tuple[1].get('res', None) or in_tuple[1].get('resolution', None)
    if res:
        out.append('-'.join(('res', str(res))))
    return '_'.join(out)


def reference2dict(in_tuple):
    """
    Split a spatial reference given as a tuple into a dictionary.

    Examples
    --------
    >>> reference2dict(('MNIPediatricAsym:cohort-2', {'res': 2}))
    {'space': 'MNIPediatricAsym', 'cohort': '2', 'resolution': '2'}

    >>> reference2dict(('MNIPediatricAsym:cohort-2', {'res': 2, 'resolution': 1}))
    {'space': 'MNIPediatricAsym', 'cohort': '2', 'resolution': '1'}

    >>> reference2dict(('MNIPediatricAsym:cohort-2', {'res': 2, 'den': '91k'}))
    {'space': 'MNIPediatricAsym', 'cohort': '2', 'resolution': '2', 'density': '91k'}

    """
    tpl_entities = ('space', 'cohort')
    retval = {tpl_entities[i]: v.split('-')[i] for i, v in enumerate(in_tuple[0].split(':'))}
    retval.update(
        {
            'resolution' if k == 'res' else 'density' if k == 'den' else k: f'{v}'
            for k, v in in_tuple[1].items()
        }
    )
    return retval


def _expand_entities(entities):
    """
    Generate multiple replacement queries based on all combinations of values.

    Ported from PyBIDS


    .. testsetup::

        >>> if PY_VERSION < (3, 6):
        ...     pytest.skip("This doctest does not work on python <3.6")

    Examples
    --------
    >>> entities = {'subject': ['01', '02'], 'session': ['1', '2'], 'task': ['rest', 'finger']}
    >>> _expand_entities(entities)  # doctest: +NORMALIZE_WHITESPACE
    [{'subject': '01', 'session': '1', 'task': 'rest'},
     {'subject': '01', 'session': '1', 'task': 'finger'},
     {'subject': '01', 'session': '2', 'task': 'rest'},
     {'subject': '01', 'session': '2', 'task': 'finger'},
     {'subject': '02', 'session': '1', 'task': 'rest'},
     {'subject': '02', 'session': '1', 'task': 'finger'},
     {'subject': '02', 'session': '2', 'task': 'rest'},
     {'subject': '02', 'session': '2', 'task': 'finger'}]

    """
    keys = list(entities.keys())
    values = list(product(*[entities[k] for k in keys]))
    return [dict(zip(keys, combs, strict=False)) for combs in values]
