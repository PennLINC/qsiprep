# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Per-task GPU selection.

Several tools QSIPrep drives ship both a CPU and a GPU build, but they disagree
on how the choice is expressed: ``eddy`` and the TORTOISE tools switch to a
differently-named executable, ``mri_synthstrip`` takes an opt-in ``-g`` flag, and
``mri_synthseg`` takes an opt-**out** ``--cpu`` flag. Historically the first two
were reachable only through per-tool JSON config files and the last two were not
reachable at all.

The trouble with a single on/off switch is that GPU memory, not the pipeline, is
usually the binding constraint: an 8 GB card comfortably runs DIFFPREP, DRBUDDI
and eddy but cannot run SynthStrip or SynthSeg. ``--gpu`` therefore takes a list
of tasks, and every construction site asks :func:`gpu_enabled` so the polarity
quirks are normalized in exactly one place.
"""

import logging
import os
import shutil
import subprocess

LOGGER = logging.getLogger('nipype.workflow')

#: Tasks that can run on a GPU, mapped to the executable that must be present.
#: ``None`` means the task uses the same executable as its CPU form and only
#: changes a flag.
GPU_TASKS = {
    'eddy': None,  # resolved at check time by _find_eddy_cuda()
    'diffprep': 'TORTOISEProcess_cuda',
    'drbuddi': 'DRBUDDI_cuda',
    'synthstrip': 'mri_synthstrip',
    'synthseg': 'mri_synthseg',
}

#: Accepted by ``--gpu`` in addition to the task names above.
GPU_ALIASES = ('all', 'none')


def resolve_gpu_tasks(requested):
    """Expand ``--gpu`` values into a concrete set of task names.

    ``none`` (or an empty list) yields an empty set; ``all`` yields every task.
    ``none`` wins over anything else so ``--gpu all none`` is unambiguously off.
    """
    requested = list(requested or [])
    if not requested or 'none' in requested:
        return set()
    if 'all' in requested:
        return set(GPU_TASKS)
    return {task for task in requested if task in GPU_TASKS}


def gpu_enabled(task, config_file_value=None):
    """Whether ``task`` should run on the GPU.

    Parameters
    ----------
    task : :obj:`str`
        One of :data:`GPU_TASKS`.
    config_file_value : :obj:`bool` or ``None``
        The legacy per-tool setting -- ``"use_cuda"`` in ``--eddy-config`` or
        ``--diffprep-config``.

    Returns
    -------
    :obj:`bool`

    Notes
    -----
    ``--gpu`` wins **when it is given**, and a disagreement is logged. When it is
    absent entirely (``config.workflow.gpu is None``) the legacy per-tool value
    still decides, so an existing ``--eddy-config`` with ``"use_cuda": true``
    keeps running on the GPU instead of silently dropping to CPU. That is the
    difference between ``--gpu none`` (explicitly off) and no ``--gpu`` at all.
    """
    from .. import config

    if task not in GPU_TASKS:
        raise ValueError(f'Unknown GPU task {task!r}; expected one of {sorted(GPU_TASKS)}')

    requested = getattr(config.workflow, 'gpu', None)
    if requested is None:
        return bool(config_file_value)

    from_cli = task in resolve_gpu_tasks(requested)

    if config_file_value is not None and bool(config_file_value) != from_cli:
        LOGGER.warning(
            'GPU setting for %s conflicts: --gpu says %s but the tool config file '
            'says use_cuda=%s. Using --gpu (%s).',
            task,
            'on' if from_cli else 'off',
            bool(config_file_value),
            'on' if from_cli else 'off',
        )
    return from_cli


def _gpu_visible():
    """Is a CUDA device actually reachable from this process?

    Inside a container the NVIDIA toolkit injects ``nvidia-smi`` only when the
    GPU was requested (``docker --gpus all`` / ``apptainer --nv``), so its
    absence is the usual symptom of a forgotten flag rather than a missing card.
    Fall back to the device nodes when ``nvidia-smi`` is unavailable but the
    driver is bind-mounted.
    """
    smi = shutil.which('nvidia-smi')
    if smi is not None:
        try:
            proc = subprocess.run(
                [smi, '-L'], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if proc.returncode == 0 and 'GPU 0' in proc.stdout:
            return True
        return False
    return os.path.exists('/dev/nvidiactl')


def _missing_binary(task):
    """Executable required by ``task`` that is not on ``PATH``, or ``None``."""
    if task == 'eddy':
        from ..interfaces.eddy import _find_eddy_cuda

        binary = _find_eddy_cuda()
    else:
        binary = GPU_TASKS[task]
    if binary is None:
        return None
    return None if shutil.which(binary) else binary


def check_gpu_available(requested):
    """Fail fast when ``--gpu`` asks for something this machine cannot deliver.

    Called at argument-parse time so a missing GPU costs seconds instead of
    surfacing as a node crash after the anatomical workflow has already run.
    Falling back to CPU silently is deliberately not an option: the CUDA and CPU
    builds are not numerically identical, so a fallback would hand back results
    from a different code path than the one that was asked for.

    Raises
    ------
    RuntimeError
        If no CUDA device is visible, or a required executable is missing.
    """
    tasks = resolve_gpu_tasks(requested)
    if not tasks:
        return

    if not _gpu_visible():
        raise RuntimeError(
            f'--gpu requested for {", ".join(sorted(tasks))}, but no CUDA device is '
            'visible to this process. Expose the GPU to the container with '
            '"docker run --gpus all" or "apptainer run --nv", or drop --gpu.'
        )

    missing = {task: _missing_binary(task) for task in sorted(tasks)}
    missing = {task: binary for task, binary in missing.items() if binary}
    if missing:
        details = '; '.join(f'{task} needs {binary}' for task, binary in missing.items())
        raise RuntimeError(
            f'--gpu requested for {", ".join(sorted(missing))}, but the GPU build is '
            f'not installed: {details}. Use a container image that ships them, or '
            'drop those tasks from --gpu.'
        )
