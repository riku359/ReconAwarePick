"""The only module that talks to CryoSPARC (cryosparc-tools).

Every call is a pattern confirmed against a live CryoSPARC v4.7.1 server:
  connect ................. CryoSPARC(license, email, password, host, base_port)  (:337)
  find_project ............ cs.find_project(pid)                                   (:361)
  create job (no queue) ... project.create_job(ws, type, params=...)              (:396)
  connect inputs .......... job.connect(slot, src_uid, src_out)                   (:411)
  set params .............. job.set_param(k, v)                                    (:468)
  queue + wait ............ job.queue(lane=, gpus=, hostname=); job.wait_for_done()(:412)
"""
from __future__ import annotations

from typing import Optional

from .config import ConnectionConfig
from .jobs._base import JobResult


class CryoSPARCApi:
    def __init__(self, conn: ConnectionConfig):
        self.conn = conn
        self.cs = None            # cryosparc.tools.CryoSPARC, set by connect()
        self.project_uid = None   # active project for create_and_wait, set by use_project()

    def use_project(self, project_uid: str) -> None:
        self.project_uid = project_uid

    # --- connection / introspection -------------------------------------
    def connect(self) -> None:
        """Open + verify the connection. Imported lazily so --help works without tools."""
        from cryosparc.tools import CryoSPARC

        self.cs = CryoSPARC(
            license=self.conn.license,
            email=self.conn.email,
            password=self.conn.password,
            host=self.conn.host,
            base_port=self.conn.port,
        )
        if not self.cs.test_connection():
            raise RuntimeError(
                f"could not reach CryoSPARC at {self.conn.host}:{self.conn.port} "
                "(check host/port and that the master is running)"
            )

    def print_job_types(self) -> None:
        self.cs.print_job_types()

    # --- project / workspace --------------------------------------------
    def find_project(self, project_uid: str):
        return self.cs.find_project(project_uid)

    def get_or_create_workspace(self, project_uid: str, title: str) -> str:
        """Return an existing workspace UID by title, else create one."""
        project = self.find_project(project_uid)
        for ws in self.cs.cli.list_workspaces(project_uid):
            if ws.get("title") == title:
                return ws["uid"]
        return project.create_workspace(title=title).uid

    # --- jobs ------------------------------------------------------------
    def create_and_wait(
        self,
        workspace_uid: str,
        name: str,
        job_type: str,
        params: dict,
        connections: dict,        # input slot -> (src_job_uid, src_output_port)
        expose: dict,             # logical output name -> output port name
        gpu: Optional[dict] = None,
        lane: str = "default",
        wait: bool = True,
    ) -> JobResult:
        """Create the job in the active project, wire inputs, set params, queue, wait."""
        project = self.find_project(self.project_uid)
        job = project.create_job(workspace_uid, job_type)
        for slot, (src_uid, src_out) in connections.items():
            job.connect(slot, src_uid, src_out)
        for key, value in params.items():
            job.set_param(key, value)

        queue_kwargs = {"lane": lane}
        if gpu and gpu.get("hostname"):
            gpus = gpu.get("gpus")
            if not gpus and gpu.get("auto"):
                # Recurrence guard for CUDA_ERROR_OUT_OF_MEMORY on shared GPUs:
                # pin to a physically-free card now instead of letting the scheduler
                # auto-place onto one another user has already filled.
                from . import gpu_select
                free = gpu_select.pick_free_gpu(cryosparcm=gpu.get("cryosparcm"))
                gpus = [free] if free is not None else None
            # Pin the worker whenever we know it -- including CPU-only jobs, where gpus
            # is None. Both worker targets sit on the 'default' lane, so an unpinned CPU
            # job (e.g. import_particles) can be scheduled onto a stale/unreachable node
            # and die at launch with ssh exit 255; pinning keeps every job on one worker.
            queue_kwargs["hostname"] = gpu["hostname"]
            if gpus:
                queue_kwargs["gpus"] = gpus
        job.queue(**queue_kwargs)

        if wait:
            job.wait_for_done()
            status = self.status(self.project_uid, job.uid)
            if status != "completed":
                raise RuntimeError(
                    f"{name} ({job.uid}, {job_type}) ended as {status!r}, not completed. "
                    f"See the job log: cryosparcm joblog {self.project_uid} {job.uid}")
        else:
            status = self.status(self.project_uid, job.uid)
        return JobResult(name=name, job_uid=job.uid, job_type=job_type,
                         status=status, outputs=dict(expose))

    def find_job(self, project_uid: str, job_uid: str):
        return self.find_project(project_uid).find_job(job_uid)

    def status(self, project_uid: str, job_uid: str) -> str:
        return self.find_job(project_uid, job_uid).doc.get("status", "unknown")

    def job_dir(self, project_uid: str, job_uid: str) -> str:
        """Absolute path to the job directory (source of truth for maps/stacks)."""
        job = self.find_job(project_uid, job_uid)
        return str(job.dir())

    # --- output readers --------------------------------------------------
    def read_gsfsc(self, project_uid: str, job_uid: str) -> dict:
        """Refined resolution (GSFSC 0.143), lower Å = better. Prefers the output
        summary res_gsfsc_tight/0143, falls back to the last progress gsfsc
        (the order CryoSPARC's own job lifecycle requires)."""
        doc = self.find_job(project_uid, job_uid).doc
        for grp in doc.get("output_result_groups", []) or []:
            summary = grp.get("summary") or {}
            for key in ("res_gsfsc_tight", "res_gsfsc_0143"):
                if summary.get(key) is not None:
                    return {"res_0143": float(summary[key])}
        for item in reversed(doc.get("progress", []) or []):
            if isinstance(item, dict) and item.get("gsfsc") is not None:
                return {"res_0143": float(item["gsfsc"])}
        return {"res_0143": None}

    def load_output(self, project_uid: str, job_uid: str, name: str, slots=None):
        job = self.find_job(project_uid, job_uid)
        # tools' load_output iterates `slots`; passing None breaks it, so omit when None.
        return job.load_output(name) if slots is None else job.load_output(name, slots=slots)

    def output_count(self, project_uid: str, job_uid: str, name: str) -> int:
        """Number of rows in an output dataset (e.g. particle count)."""
        return len(self.load_output(project_uid, job_uid, name))

    # --- job assets (CryoSPARC-rendered GUI plots) -----------------------
    def list_job_assets(self, project_uid: str, job_uid: str) -> list:
        """GridFS asset metadata for a job -- the PNG/PDF plots CryoSPARC renders
        in the GUI (FSC curve, viewing-direction distribution, ...). Each entry is
        a dict with `_id` (fileid), `filename`, `contentType`, `length`."""
        return self.find_job(project_uid, job_uid).list_assets()

    def download_asset(self, fileid: str, target):
        """Download one GridFS asset by fileid to a file path or directory."""
        return self.cs.download_asset(fileid, target)
