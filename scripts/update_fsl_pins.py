#!/usr/bin/env python
"""Pin the FSL conda packages in pyproject.toml to a published FSL release.

FSL releases are defined by conda environment files published alongside a
manifest at https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/. This
script reads the environment file for a given release and rewrites the version
of every ``fsl*`` package in the pixi dependency tables of pyproject.toml so
that the packages we install come from a single, coherent FSL release rather
than from whatever the solver happens to pick.

Run ``pixi lock`` afterwards to update pixi.lock.

Examples
--------
List the releases the manifest knows about::

    python scripts/update_fsl_pins.py --list

Pin to a specific release::

    python scripts/update_fsl_pins.py 6.0.7.23

Fail (without editing anything) if the pins have drifted, for use in CI::

    python scripts/update_fsl_pins.py 6.0.7.23 --check
"""

import argparse
import difflib
import json
import re
import sys
import urllib.request
from pathlib import Path

MANIFEST_URL = 'https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/manifest.json'
CHANNEL_URL = 'https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public'
DEFAULT_PLATFORM = 'linux-64'
# Manifest platform names are not conda subdir names.
CONDA_SUBDIRS = {
    'linux-64': 'linux-64',
    'linux-aarch64': 'linux-aarch64',
    'macos-64': 'osx-64',
    'macos-M1': 'osx-arm64',
}

# ``[tool.pixi.dependencies]``, ``[tool.pixi.feature.<name>.dependencies]``, ...
# but not ``[tool.pixi.feature.<name>.pypi-dependencies]``.
CONDA_TABLE_RE = re.compile(r'tool\.pixi\..*(?<!pypi-)dependencies')
TABLE_RE = re.compile(r'^\[(?P<name>[^\]]+)\]')
ENTRY_RE = re.compile(
    r'^(?P<key>"[^"]+"|[^\s=#]+)(?P<sep>\s*=\s*)(?P<value>.+?)(?P<comment>\s+#.*)?$'
)
VERSION_IN_TABLE_RE = re.compile(r'(version\s*=\s*)"[^"]*"')
# ``fsl-eddy-cuda-11.0`` is the CUDA 11 build of the release's ``fsl-eddy-cuda``.
CUDA_VARIANT_RE = re.compile(r'^(?P<base>.+-cuda)(-\d+(\.\d+)*)?$')


def fetch(url):
    """Download a text resource over HTTPS."""
    if not url.startswith('https://'):
        raise ValueError(f'Refusing to fetch non-HTTPS URL: {url}')
    with urllib.request.urlopen(url) as response:  # noqa: S310 - scheme checked above
        return response.read().decode('utf-8')


def release_environment_url(manifest, version, platform):
    """Find the conda environment file for one release and platform."""
    versions = manifest['versions']
    if version not in versions:
        raise SystemExit(
            f'FSL release {version} is not in the manifest. '
            f'Known releases: {", ".join(sorted_versions(manifest))}'
        )
    # ``latest`` is an alias for a real release.
    while isinstance(versions[version], str):
        version = versions[version]

    for entry in versions[version]:
        if entry.get('platform') == platform:
            return version, entry['environment']

    platforms = sorted(entry.get('platform', '?') for entry in versions[version])
    raise SystemExit(f'FSL {version} has no {platform} build. Available: {", ".join(platforms)}')


def sorted_versions(manifest):
    """Manifest release names, oldest first, with ``latest`` last."""

    def key(version):
        # Numeric components sort numerically and before any non-numeric ones.
        return [(0, int(p), '') if p.isdigit() else (1, 0, p) for p in version.split('.')]

    versions = sorted((name for name in manifest['versions'] if name != 'latest'), key=key)
    return [*versions, 'latest']


def parse_environment(text):
    """Map package name to version for the ``dependencies`` list of an environment file.

    The published files are flat lists of ``- name version [build]`` entries, so
    they are parsed directly instead of pulling in a YAML dependency.
    """
    packages = {}
    in_dependencies = False
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith('#'):
            continue
        if not line[0].isspace():
            in_dependencies = entry == 'dependencies:'
            continue
        if not in_dependencies or not entry.startswith('- '):
            continue
        fields = entry[2:].split()
        if len(fields) > 1:
            packages[fields[0]] = fields[1]
    return packages


def channel_versions(platform):
    """Map package name to the set of versions published in the FSL channel."""
    versions = {}
    for subdir in (CONDA_SUBDIRS[platform], 'noarch'):
        repodata = json.loads(fetch(f'{CHANNEL_URL}/{subdir}/repodata.json'))
        for group in ('packages', 'packages.conda'):
            for info in repodata.get(group, {}).values():
                versions.setdefault(info['name'], set()).add(info['version'])
    return versions


def release_version(packages, name):
    """Version of ``name`` in the release, allowing for CUDA-variant renames."""
    if name in packages:
        return packages[name], name

    variant = CUDA_VARIANT_RE.match(name)
    if variant:
        base = variant.group('base')
        for candidate, version in packages.items():
            match = CUDA_VARIANT_RE.match(candidate)
            if match and match.group('base') == base:
                return version, candidate
    return None, None


def update_pyproject(text, packages, available=None):
    """Rewrite the pixi FSL pins. Returns the new text plus notes and warnings.

    ``packages`` maps package name to version for the target release and
    ``available``, if given, maps package name to the versions published in
    the FSL channel. Pins that the channel cannot satisfy are skipped, since
    FSL occasionally renames a package (the CUDA builds of eddy, for instance)
    and a pin on the old name would make the environment unsolvable.
    """
    lines = text.splitlines(keepends=True)
    in_conda_table = False
    notes = []
    warnings = []

    for index, line in enumerate(lines):
        table = TABLE_RE.match(line)
        if table:
            in_conda_table = bool(CONDA_TABLE_RE.fullmatch(table.group('name')))
            continue
        if not in_conda_table:
            continue

        entry = ENTRY_RE.match(line.rstrip('\n'))
        if not entry:
            continue
        name = entry.group('key').strip('"')
        if not name.startswith('fsl'):
            continue

        version, source = release_version(packages, name)
        if version is None:
            warnings.append(f'{name} is not part of this FSL release; leaving it alone')
            continue
        if available is not None and version not in available.get(name, ()):
            replacement = f'; the release ships {source} {version}' if source != name else ''
            warnings.append(
                f'{name} {version} is not published in the FSL channel{replacement}. '
                'Leaving it alone -- the dependency probably needs to be renamed.'
            )
            continue
        if source != name:
            notes.append(f'{name} pinned from the release entry for {source}')

        value = entry.group('value')
        if value.startswith('{'):
            new_value, count = VERSION_IN_TABLE_RE.subn(rf'\g<1>"=={version}"', value)
            if not count:
                warnings.append(f'{name} has no version field in its table; leaving it alone')
                continue
        else:
            new_value = f'"=={version}"'

        lines[index] = (
            f'{entry.group("key")}{entry.group("sep")}{new_value}{entry.group("comment") or ""}\n'
        )

    return ''.join(lines), notes, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        'version',
        nargs='?',
        default='latest',
        help='FSL release to pin to, e.g. 6.0.7.23 (default: %(default)s)',
    )
    parser.add_argument(
        '--platform',
        default=DEFAULT_PLATFORM,
        help='Platform whose environment file is read (default: %(default)s)',
    )
    parser.add_argument(
        '--pyproject',
        type=Path,
        default=Path(__file__).parent.parent / 'pyproject.toml',
        help='Path to pyproject.toml (default: %(default)s)',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Report needed changes and exit non-zero instead of writing them',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List the FSL releases in the manifest and exit',
    )
    args = parser.parse_args()

    manifest = json.loads(fetch(MANIFEST_URL))
    if args.list:
        print('\n'.join(sorted_versions(manifest)))
        return 0

    version, url = release_environment_url(manifest, args.version, args.platform)
    print(f'Reading FSL {version} ({args.platform}) from {url}')
    packages = parse_environment(fetch(url))

    available = None
    if args.platform in CONDA_SUBDIRS:
        available = channel_versions(args.platform)

    text = args.pyproject.read_text()
    updated, notes, warnings = update_pyproject(text, packages, available)

    for note in notes:
        print(f'note: {note}')
    for warning in warnings:
        print(f'warning: {warning}', file=sys.stderr)

    if updated == text:
        print(f'{args.pyproject} already matches FSL {version}')
        return 0

    diff = difflib.unified_diff(
        text.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(args.pyproject),
        tofile=f'{args.pyproject} (FSL {version})',
    )
    print(''.join(diff), end='')

    if args.check:
        print(f'{args.pyproject} does not match FSL {version}', file=sys.stderr)
        return 1

    args.pyproject.write_text(updated)
    print(f'Pinned {args.pyproject} to FSL {version}. Run `pixi lock` to update pixi.lock.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
