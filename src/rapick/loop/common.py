"""Shared machinery for the feedback-loop drivers.

A driver is a long chain of shell-outs that must survive being killed: a step runs for
tens of minutes to hours, so "record every step, skip what is recorded" is not a
convenience but the only way a chain finishes at all on a host that gets pre-empted,
rebooted, or otherwise interrupted mid-round.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# flock, not a lock file: the kernel releases it when the holder dies, so a killed
# driver never leaves a stale lock behind. It has to sit on a local filesystem --
# flock over NFS is advisory at best, and two hosts would both think they hold it.
LOCK_DIR = Path(os.environ.get("RAPICK_LOCK_DIR") or "/tmp")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd, log_path: Path, env_extra=None):
    """Run a step, tee-ing to log_path. Raises on non-zero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **(env_extra or {})}
    log(f"$ (cd {cwd}) {' '.join(str(c) for c in cmd)}")
    with log_path.open("w") as fh:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_lines = []
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            out_lines.append(line)
        proc.wait()
    if proc.returncode != 0:
        tail = "".join(out_lines[-25:])
        raise RuntimeError(f"step failed (exit {proc.returncode}); log {log_path}\n{tail}")
    return "".join(out_lines)


def acquire_lock(path: Path, what: str):
    """Refuse to start a second driver against the same GPU and CryoSPARC project.

    Two instances share the GPU, the project and -- when they run the same arm -- the
    same state.json, and the damage is quiet: one recorded overlap ran a round's
    fine-tune twice, wrote that round's model twice, and built the next round an
    extract and a class_2D per process, with two 2D-selection cycles underneath.
    Nothing failed; the records simply stopped describing the files on disk.

    Returns the open handle, which the caller must keep alive for the process's
    lifetime. flock releases automatically if the holder dies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another {what} holds {path}; refusing to run a second one against the "
                 f"same GPU and CryoSPARC project")
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


class State:
    """Per-run progress, so a killed driver resumes instead of rebuilding."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.is_file() else {}

    def done(self, step):
        return step in self.data

    def get(self, step, key=None, default=None):
        rec = self.data.get(step, {})
        return rec if key is None else rec.get(key, default)

    def mark(self, step, **payload):
        self.data[step] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


def wait_for_free_gpu(index, min_free_mb: int, max_wait_s: int) -> None:
    """Block until GPU `index` has room, or until the deadline passes.

    Starting a fine-tune on a card somebody else has filled does not fail cheaply: it
    OOMs minutes in, and the round has to be redone from its `pick` step. Waiting for
    the card costs nothing by comparison, and the shared GPUs always free up.

    Best-effort by design. If the free memory cannot be read at all -- no nvidia-smi,
    a driver that will not answer -- this returns immediately and lets the caller try,
    which is what the original single-host driver did.
    """
    try:
        from rapick.recon.gpu_select import pick_free_gpu
    except ImportError:
        log("  cannot import rapick.recon.gpu_select; starting without a GPU check")
        return
    chosen = pick_free_gpu(min_free_mb=min_free_mb, candidates=[int(index)],
                           max_wait_s=max_wait_s)
    if chosen is None:
        log(f"  GPU {index} never reported {min_free_mb} MiB free within "
            f"{max_wait_s // 60} min; starting anyway")


def connect_cryosparc(env: dict):
    """Open a cryosparc-tools session from the repository-root `.env`'s credentials.

    Imported here rather than at module scope so that the reporting tools, which need
    no server at all, stay importable with nothing but the standard library.
    """
    from cryosparc.tools import CryoSPARC

    missing = [k for k in ("CRYOSPARC_LICENSE_ID", "CRYOSPARC_EMAIL",
                           "CRYOSPARC_PASSWORD") if not env.get(k)]
    if missing:
        sys.exit(f"error: {', '.join(missing)} missing from the repository-root .env")
    cs = CryoSPARC(license=env["CRYOSPARC_LICENSE_ID"], email=env["CRYOSPARC_EMAIL"],
                   password=env["CRYOSPARC_PASSWORD"],
                   host=env.get("CRYOSPARC_HOST", "localhost"),
                   base_port=int(env.get("CRYOSPARC_PORT", "39000")))
    if not cs.test_connection():
        sys.exit("error: cannot reach CryoSPARC")
    return cs


def parse_rounds(spec: str) -> list:
    """"0-3" -> [0, 1, 2, 3];  "0,2" -> [0, 2];  "2" -> [2]."""
    if "-" in spec:
        first, last = (int(v) for v in spec.split("-", 1))
        return list(range(first, last + 1))
    return [int(v) for v in spec.split(",")]
