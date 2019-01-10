
"""Handle merging and spliting of DSI files."""
import numpy as np
import os
import nibabel as nb
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, OutputMultiObject, traits, isdefined)
from nipype.interfaces import afni, ants
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.ants.resampling import ApplyTransformsInputSpec
from tempfile import TemporaryDirectory
import logging
from dipy.sims.voxel import all_tensor_evecs
from dipy.reconst.dti import decompose_tensor
from dipy.core.geometry import normalized_vector, decompose_matrix
import pandas as pd
from sklearn.metrics.regression import r2_score


class FODtoFIBGZInputSpec(BaseInterfaceInputSpec):
    fod_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
