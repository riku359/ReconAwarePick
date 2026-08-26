"""One thin module per CryoSPARC job.

Each module exposes `run(api, profile, inputs, params, workspace_uid, gpu) -> JobResult`
and documents, in its docstring, exactly which job_type it creates, which inputs
it wires, which params it sets, and which outputs it exposes. Open the module to
understand that stage; the job-type/port values themselves live in
configs/profiles/cryosparc_v47.yaml (single source of truth).
"""
from . import (
    abinit_reconstruction,
    classification_2d,
    ctf_estimation,
    homogeneous_refinement,
    import_micrographs,
    import_particles,
    junk_detector,
    local_resolution,
    orientation_diagnosis,
    particle_extraction,
)
from ._base import JobResult, JobSpec

__all__ = [
    "JobResult",
    "JobSpec",
    "import_micrographs",
    "ctf_estimation",
    "import_particles",
    "particle_extraction",
    "junk_detector",
    "classification_2d",
    "abinit_reconstruction",
    "homogeneous_refinement",
    "local_resolution",
    "orientation_diagnosis",
]
