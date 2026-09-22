"""Loading the effective TORTOISE DIFFPREP configuration.

``--diffprep-config`` is read by ``init_diffprep_hmc_wf``, to build DIFFPREP,
and by :mod:`qsiprep.utils.jacobian_provenance`, to decide whether the
eddy-current Jacobian applies. Both load it here so they cannot disagree about
what the file says or which defaults fill it in.
"""

import json
from importlib.resources import files

from .resources import as_path


def load_diffprep_config(config_path):
    """Load a --diffprep-config JSON, or return defaults."""
    if config_path is None:
        config_path = as_path(files('qsiprep.data') / 'diffprep_params.json')
    with open(config_path) as fobj:
        cfg = json.load(fobj)
    cfg.setdefault('b0_id', -1)
    cfg.setdefault('is_human_brain', True)
    cfg.setdefault('rot_eddy_center', 'isocenter')
    cfg.setdefault('extra_args', [])
    # --hmc-method exposes a single "tortoise" value, so this is the only way to
    # reach DIFFPREP's rigid-only ('motion') or 'cubic' eddy modes.
    cfg.setdefault('correction_mode', 'quadratic')
    # No default for "use_cuda": its absence must stay observable so a shipped
    # default is never mistaken for user intent (see
    # qsiprep.workflows.dwi.diffprep._legacy_use_cuda).
    # Opt-in MAPMRI shell synthesis for DRBUDDI's registration target;
    # None/0 = off.
    cfg.setdefault('drbuddi_synth_shell_bval', None)
    cfg.setdefault('drbuddi_synth_shell_ndirs', 30)
    return cfg
