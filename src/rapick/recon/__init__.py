"""rapick.recon: explicit, config-driven CryoSPARC v4.7 reconstruction pipeline.

Drives one fixed job chain per source, job by job:

    import_micrographs -> patch_ctf -> import_particles -> extract -> class_2D
      -> homo_abinit x3 seeds -> homo_refine x3 seeds
      -> best-of-3 by GSFSC 0.143 -> local_resolution -> collect (metrics.json)

Everything about *what* runs -- job types, ports, params, sources, seeds -- lives in
configs/. Nothing is hardcoded in the job wrappers. See README.md in this package and
docs/CRYOSPARC.md at the repository root.
"""

__version__ = "0.1.0"
