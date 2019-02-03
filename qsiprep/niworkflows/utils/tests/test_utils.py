"""Test utils"""
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import check_call
from qsiprep.niworkflows.utils.misc import _copy_any


def test_copy_gzip():
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        filepath = tmppath / 'name1.txt'
        filepath2 = tmppath / 'name2.txt'
        assert not filepath2.exists()
        open(str(filepath), 'w').close()
        check_call(['gzip', '-N', str(filepath)])
        assert not filepath.exists()

        gzpath1 = '%s/%s' % (tmppath, 'name1.txt.gz')
        gzpath2 = '%s/%s' % (tmppath, 'name2.txt.gz')
        _copy_any(gzpath1, gzpath2)
        assert Path(gzpath2).exists()
        check_call(['gunzip', '-N', '-f', gzpath2])
        assert not filepath.exists()
        assert filepath2.exists()
