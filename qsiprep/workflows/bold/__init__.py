# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=unused-import
"""

Pre-processing fMRI - BOLD signal workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qsiprep.workflows.bold.base
.. automodule:: qsiprep.workflows.bold.util
.. automodule:: qsiprep.workflows.bold.hmc
.. automodule:: qsiprep.workflows.bold.stc
.. automodule:: qsiprep.workflows.bold.t2s
.. automodule:: qsiprep.workflows.bold.registration
.. automodule:: qsiprep.workflows.bold.resampling
.. automodule:: qsiprep.workflows.bold.confounds


"""

from .base import init_func_preproc_wf
from .util import init_bold_reference_wf
from .hmc import init_bold_hmc_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf
from .registration import (
    init_bold_t1_trans_wf,
    init_bold_reg_wf,
)
from .resampling import (
    init_bold_mni_trans_wf,
    init_bold_surf_wf,
    init_bold_preproc_trans_wf,
)

from .confounds import (
    init_bold_confs_wf,
    init_ica_aroma_wf,
)
