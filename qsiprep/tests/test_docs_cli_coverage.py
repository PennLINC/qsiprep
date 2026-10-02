"""The running page documents every ``--ignore`` and ``--force`` value.

The page keeps hand-written tables of these values because it says what each
one changes downstream, which the parser help does not. This test keeps them
from drifting apart when a value is added or renamed.
"""

from pathlib import Path

import pytest

RUNNING_PAGE = Path(__file__).resolve().parents[2] / 'docs' / 'running.rst'


@pytest.mark.parametrize('dest', ['ignore', 'force'])
def test_running_page_lists_every_choice(dest):
    if not RUNNING_PAGE.exists():
        pytest.skip('docs are not part of the installed package')
    from qsiprep.cli.parser import _build_parser

    action = next(a for a in _build_parser()._actions if a.dest == dest)
    text = RUNNING_PAGE.read_text()
    missing = [choice for choice in action.choices if f'``{choice}``' not in text]
    assert not missing, f'--{dest} values missing from docs/running.rst: {missing}'
