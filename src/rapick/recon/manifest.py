"""Experiment-record manifest (NOT a cache).

One manifest per (dataset, setting, source); a shared one per (dataset, setting)
for the reused import+ctf jobs. Stored under the artifact tree so it is a
first-class, kept record of provenance:

  ${RAPICK_WORK}/empiar_<id>/<setting>/<source>/manifest.json
  ${RAPICK_WORK}/empiar_<id>/<setting>/_shared/manifest.json

Resume reads recorded job UIDs (via api.find_job) and reuses a step only if its
job is completed with outputs present. It never scans the workspace for the
"latest completed job".

A recorded `completed` status is evidence the job finished, NOT evidence it was
correct: a step that ran over an incomplete micrograph set stays "done" forever.
Patch CTF has been observed reporting `completed` while silently dropping
micrographs. Verify micrograph counts and STAR hashes before spending GPU time on
top of a reused step, and never auto-continue past a preflight failure.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .jobs._base import JobResult


@dataclass
class TrialRecord:
    seed: int
    abinit: Optional[str] = None
    refine: Optional[str] = None
    res_0143: Optional[float] = None


@dataclass
class Manifest:
    dataset: str
    setting: str
    source: str
    project_uid: Optional[str] = None
    workspace_uid: Optional[str] = None
    shared: dict = field(default_factory=dict)   # step -> {uid, job_type, outputs, status}
    jobs: dict = field(default_factory=dict)     # step -> {uid, job_type, outputs, status}
    trials: list = field(default_factory=list)
    best_seed: Optional[int] = None
    local_res: Optional[str] = None
    input_star: dict = field(default_factory=dict)
    status: str = "pending"


def path_for(work_root: str, dataset: str, setting: str, source: str) -> Path:
    """Manifest path for one source (use source='_shared' for import+ctf).
    `work_root` is $RAPICK_WORK."""
    return Path(work_root) / dataset / setting / source / "manifest.json"


def load(path: str | Path) -> Optional[Manifest]:
    path = Path(path)
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    d["trials"] = [TrialRecord(**t) for t in d.get("trials", [])]
    return Manifest(**d)


def save(manifest: Manifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=False))


def record(manifest_jobs: dict, step: str, result: JobResult,
           input_fingerprint: Optional[str] = None) -> None:
    """Store a JobResult into a manifest `shared`/`jobs` dict. `input_fingerprint`
    identifies the inputs the job ran on (star hash, micrograph-set hash) so a later
    run can tell whether those inputs have since changed."""
    manifest_jobs[step] = {
        "uid": result.job_uid,
        "job_type": result.job_type,
        "outputs": result.outputs,
        "status": result.status,
        "input_fingerprint": input_fingerprint,
    }


def reuse_or_none(api, manifest_jobs: dict, step: str,
                  input_fingerprint: Optional[str] = None) -> Optional[JobResult]:
    """Return a JobResult for a recorded step iff its job is completed AND its inputs
    are unchanged, else None (so the caller rebuilds it).

    A completed job is refused when the caller supplies an `input_fingerprint` that
    differs from the one recorded — this is what stops a stale job (e.g. a picker's
    star replaced by a cleaned version) from being silently reused. A record with no
    fingerprint (written before this field existed) cannot be verified, so it is
    reused for backward compatibility; `_step` warns when that happens."""
    rec = manifest_jobs.get(step)
    if not rec:
        return None
    if api.status(api.project_uid, rec["uid"]) != "completed":
        return None
    recorded = rec.get("input_fingerprint")
    if input_fingerprint is not None and recorded is not None and recorded != input_fingerprint:
        return None
    return JobResult(name=step, job_uid=rec["uid"], job_type=rec["job_type"],
                     status="completed", outputs=rec.get("outputs", {}))


def star_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
