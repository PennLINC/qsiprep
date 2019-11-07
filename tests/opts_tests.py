import pytest
import os
from qsiprep.cli.run import set_freesurfer_license, validate_bids, get_parser

base_args = "bids out participant --output_resolution 2.3"


def test_required():
    """error if we dont have necessary args"""
    args = base_args.split(' ')

    # all we need
    opts = get_parser().parse_args(args)

    # none
    with pytest.raises(SystemExit) as pa_fail:
        get_parser().parse_args([])
    assert pa_fail.value.code == 2

    # missing one of any of the base args
    for part_args in ["bids", "bids out", "bid out participant"]:
        with pytest.raises(SystemExit) as pa_fail:
            get_parser().parse_args(part_args.split(' '))
        assert pa_fail.value.code == 2


def test_required_recononly(monkeypatch):
    # dont need output_resolution if we have recon_only
    base_args = "bids out participant --recon_only"
    args = base_args.split(' ')
    # sys.argv used to set if output-res required
    monkeypatch.setattr('qsiprep.cli.run.sys.argv', args)
    get_parser().parse_args(args)


def test_set_freesurfer_license(tmpdir):
    """test setting, precedence, and error if DNE"""
    # create temp file
    lic1 = tmpdir.join("license.txt")
    lic2 = tmpdir.join("license2.txt")
    lic3 = tmpdir.join("license3.txt")
    for lic in [lic1, lic2, lic3]:
        lic.write('not a real license')
    opts = get_parser().parse_args(base_args.split(' '))

    # no FS file, throw an error
    if os.getenv('FREESURFER_HOME'):
        del os.environ['FREESURFER_HOME']
    if os.getenv('FS_LICENSE'):
        del os.environ['FS_LICENSE']
    with pytest.raises(RuntimeError):
        fs = set_freesurfer_license(opts)

    # using FRESURFER_HOME
    os.environ['FREESURFER_HOME'] = tmpdir.__str__()
    set_freesurfer_license(opts)
    assert os.getenv('FS_LICENSE') == f'{lic1}'

    # set directly -- check function output
    os.environ['FS_LICENSE'] = f'{lic2}'
    assert set_freesurfer_license(opts) == f'{lic2}'

    # using command line switch overwrites other options
    fsarg = f"{base_args} --fs-license-file {lic3}"
    opts = get_parser().parse_args(fsarg.split(' '))
    set_freesurfer_license(opts)
    assert os.getenv('FS_LICENSE') == f'{lic3}'


@pytest.mark.parametrize("will_validate,opts_str", (
  (True, base_args),                                           # run if base args
  (False, base_args + " --skip_bids_validation"),              # not if skipped
  (False, base_args + " --recon-only"),                        # or recon all
  (False, base_args + " --skip_bids_validation --recon-only")  # or both
))
def test_validate_bids(monkeypatch, opts_str, will_validate):
    # from ..utils.bids import validate_input_dir
    monkeypatch.setattr("qsiprep.utils.bids.validate_input_dir", lambda *kargs: True)
    opts = get_parser().parse_args(opts_str.split(' '))
    assert will_validate == validate_bids(opts)
