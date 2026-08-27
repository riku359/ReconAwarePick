"""Pick a physically-free GPU at queue time.

The host's GPUs are shared with other users, but CryoSPARC's scheduler only knows
about GPUs reserved by its *own* jobs. Left to auto-place on the "default" lane, it
can put a job on a card another user has already filled, and the job dies with
``CUDA_ERROR_OUT_OF_MEMORY`` mid-run. For extraction that is worse than a crash: the
affected micrographs are marked incomplete and the job still reports "completed", so
particles silently go missing (one observed extract job lost 243 of its 300
micrographs this way).

`pick_free_gpu()` reads real free memory from ``nvidia-smi`` and returns the emptiest
card that clears a threshold, so a GPU job can be pinned to a card that actually has
room. Reservation exclusion via ``cryosparcm`` is best-effort and fail-open.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Iterable, Optional

# Recon jobs need ~12GB. On this heavily-shared host, requiring only 12GB free lets a
# job grab a card that's momentarily free, then OOM when a neighbour's next allocation
# lands (a race). Require 16GB free so there's a ~4GB cushion and we only run on a card
# a neighbour has genuinely (mostly) vacated — otherwise wait (see DEFAULT_MAX_WAIT_S).
DEFAULT_MIN_FREE_MB = 16000
# A card another user is actively computing on reports high utilization even while it
# still shows free memory; pinning there races the neighbour's next allocation and OOMs.
DEFAULT_MAX_UTIL = 35
# During heavy contention no card clears the thresholds; rather than pin onto a busy card
# and die, wait for one to free up (this is a multi-day batch — a wait beats a crash, and
# the shared GPUs always free up eventually as other users' jobs cycle / overnight).
DEFAULT_MAX_WAIT_S = 86400   # 24 h
POLL_S = 30


def _reserved(cryosparcm: Optional[str]) -> "set[int]":
    """GPU indices held by a running CryoSPARC job on this instance (best-effort)."""
    if not cryosparcm:
        return set()
    reserved: set[int] = set()
    try:
        running = subprocess.run(
            [cryosparcm, "cli", "get_jobs_by_status(status='running')"],
            capture_output=True, text=True, timeout=60).stdout
        for job_uid in set(re.findall(r"'uid': '(J\d+)'", running)):
            project = re.search(
                r"'uid': '" + job_uid + r"'[^}]*?'project_uid': '(P\d+)'", running)
            if not project:
                # No project uid to query with. Skipping this job only makes the
                # exclusion less complete, and this whole step is best-effort --
                # guessing a uid would query someone else's project instead.
                continue
            doc = subprocess.run(
                [cryosparcm, "cli",
                 f"get_job(project_uid='{project.group(1)}', "
                 f"job_uid='{job_uid}', *['resources_allocated'])"],
                capture_output=True, text=True, timeout=60).stdout
            slot = re.search(
                r"'resources_allocated':\s*\{.*?'slots':\s*\{[^}]*?'GPU':\s*\[([^\]]*)\]",
                doc)
            if slot:
                reserved.update(int(n) for n in re.findall(r"\d+", slot.group(1)))
    except Exception:
        return set()
    return reserved


def _gpu_stats() -> "dict[int, tuple[int, int]]":
    """GPU index -> (free MiB, utilization %). Empty dict if nvidia-smi is absent."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return {}
    stats = {}
    for line in out.strip().splitlines():
        index, free_mb, util = (field.strip() for field in line.split(","))
        stats[int(index)] = (int(free_mb), int(util))
    return stats


def pick_free_gpu(
    min_free_mb: int = DEFAULT_MIN_FREE_MB,
    candidates: Optional[Iterable[int]] = None,
    cryosparcm: Optional[str] = None,
    max_util: int = DEFAULT_MAX_UTIL,
    max_wait_s: int = DEFAULT_MAX_WAIT_S,
) -> Optional[int]:
    """Index of the emptiest GPU with >= `min_free_mb` free AND util <= `max_util`.

    On a shared host the memory check alone isn't enough: a card can show free memory
    while a neighbour is mid-computation on it, and pinning there OOMs when the
    neighbour's next allocation lands. So also skip busy cards, and — since this is an
    overnight batch — poll up to `max_wait_s` for a card to free up rather than crash.
    After the deadline, fall back to the emptiest card >= `min_free_mb` ignoring util
    (best-effort), else None (let CryoSPARC auto-place)."""
    reserved = _reserved(cryosparcm)
    waited = 0
    while True:
        stats = _gpu_stats()
        ranked = sorted(
            ((free_mb, util, index) for index, (free_mb, util) in stats.items()
             if (candidates is None or index in set(candidates)) and index not in reserved),
            reverse=True)
        for free_mb, util, index in ranked:
            if free_mb >= min_free_mb and util <= max_util:
                return index
        # deadline check runs even when nvidia-smi returned nothing, so a persistent
        # nvidia-smi failure can never hang the caller forever.
        if waited >= max_wait_s:
            for free_mb, util, index in ranked:
                if free_mb >= min_free_mb:
                    return index
            return None
        time.sleep(POLL_S)
        waited += POLL_S


def pick_free_gpus(
    n: int,
    min_free_mb: int = DEFAULT_MIN_FREE_MB,
    cryosparcm: Optional[str] = None,
    max_util: int = DEFAULT_MAX_UTIL,
) -> "list[int]":
    """Up to `n` physically-free GPU indices (emptiest first), each with >= `min_free_mb`
    free and util <= `max_util`.

    Non-blocking and best-effort: returns fewer than `n` (possibly empty) when that many
    aren't free right now. Unlike pick_free_gpu(), it never waits -- extra cards here only
    buy read parallelism for the I/O-bound extract step, so running on whatever is free
    beats stalling a multi-day batch to hold out for a full set."""
    reserved = _reserved(cryosparcm)
    ranked = sorted(
        ((free_mb, util, index) for index, (free_mb, util) in _gpu_stats().items()
         if index not in reserved),
        reverse=True)
    return [index for free_mb, util, index in ranked
            if free_mb >= min_free_mb and util <= max_util][:n]
