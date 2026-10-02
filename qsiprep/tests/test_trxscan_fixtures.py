"""The TRXScan fixture spec file in .circleci must match the recipes.

The CI cache key is that file's checksum, so it has to change whenever a recipe does.
"""

from pathlib import Path

from qsiprep.tests import trxscan_fixtures


def test_spec_file_matches_the_recipes():
    spec = Path(__file__).resolve().parents[2] / '.circleci' / 'trxscan_fixtures.txt'
    assert spec.read_text() == trxscan_fixtures.spec_text(), (
        'regenerate with: python -m qsiprep.tests.trxscan_fixtures --spec '
        '> .circleci/trxscan_fixtures.txt'
    )


def test_recipe_args_are_cli_ready():
    args = trxscan_fixtures.recipe_args('phasediff')
    assert args[0] == 'phasediff_fieldmap'
    assert '--set' in args
    assert 'anat_offset=[3, -4, 2, 5, -3, 4]' in args
    assert all(
        k in trxscan_fixtures.RECIPES
        for k in ('rpe', 'epi', 'phasediff', 'gnl', 'motion', 'offsets')
    )
