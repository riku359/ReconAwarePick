"""Building CryoSPARC select_2D / class_2D jobs from CryoSift scores.

The single-shot `purify_class2d.py` and the iterative `iterate_class2d.py` share these
steps. Class selection follows upstream (Magellon's `cryosparc_utils.auto_select`):
interact("get_class_info") -> set_class_selected -> interact("finish").
"""

import time
from dataclasses import dataclass, field

# Selection is always expressed as "keep score < cutoff". Upstream uses the same
# direction, and it follows straight from the score definition: 1.0 is the best class
# and 5.0 the worst.
TERMINAL_STATUSES = ("completed", "killed", "failed")


@dataclass
class Selection:
    """What one select_2D kept and what it dropped."""

    kept_classes: list = field(default_factory=list)
    dropped_classes: list = field(default_factory=list)
    kept_particles: int = 0
    dropped_particles: int = 0

    @property
    def total_particles(self):
        return self.kept_particles + self.dropped_particles

    def summary(self):
        total = self.total_particles
        frac = self.kept_particles / total if total else 0.0
        return (f"{len(self.kept_classes)}/{len(self.kept_classes) + len(self.dropped_classes)} classes, "
                f"{self.kept_particles:,}/{total:,} particles ({frac:.1%})")


def find_completed_class2d(project, job_uid):
    """Return a completed job usable as a class_2D, or stop right here."""
    job = project.find_job(job_uid)
    job_type = job.doc.get("type") or job.doc.get("job_type")
    if job_type not in ("class_2D", "class_2D_new"):
        raise SystemExit(f"{job_uid} is not a class_2D: {job_type}")
    if job.doc.get("status") != "completed":
        raise SystemExit(f"{job_uid} is not completed: {job.doc.get('status')}")
    return job


def parent_uids(project, job_uid):
    """Return the uids of the jobs wired into this job's inputs."""
    doc = project.find_job(job_uid).doc
    return [conn["job_uid"]
            for group in doc.get("input_slot_groups") or []
            for conn in group.get("connections") or []
            if conn.get("job_uid")]


def create_select(workspace, class2d_uid, title):
    """Create a select_2D over all classes of a class_2D and advance it to waiting.

    Waiting is the state in which classes can be selected.
    """
    job = workspace.create_job(
        "select_2D",
        title=title,
        connections={
            "particles": (class2d_uid, "particles"),
            "templates": (class2d_uid, "class_averages"),
        },
    )
    job.queue()
    job.wait_for_status("waiting")
    return job


def select_classes(job, class_scores, keep):
    """On a waiting select_2D, select exactly the classes for which keep(score) holds.

    `class_scores` maps class_idx -> score. The class_idx returned by get_class_info is
    used as the key directly, so the input must always be the full set of class_2D
    classes and never a subset.
    """
    selection = Selection()
    for class_info in job.interact("get_class_info"):
        class_idx = class_info["class_idx"]
        score = class_scores.get(class_idx)
        if score is None:
            raise SystemExit(f"no score for class {class_idx} ({len(class_scores)} classes)")

        if keep(score):
            selection.kept_classes.append(class_idx)
            selection.kept_particles += class_info["num_particles_total"]
            job.interact("set_class_selected", {"class_idx": class_idx, "selected": True})
        else:
            selection.dropped_classes.append(class_idx)
            selection.dropped_particles += class_info["num_particles_total"]

    return selection


def finish_select(job):
    job.interact("finish")
    job.wait_for_done()
    job.refresh()
    return job.doc.get("status")


def create_class2d(workspace, sources, params, title, job_type="class_2D"):
    """Create a class_2D in the building state over a set of particle output ports.

    `sources` is [(job_uid, output_name), ...]. Passing more than one merges those
    particle sets, which is what the paper's final classification does when it brings
    the loop survivors back together with the set-aside classes.
    """
    return workspace.create_job(
        job_type,
        title=title,
        connections={"particles": list(sources)},
        params=dict(params),
    )


def queue_and_wait(job, lane=None, hostname=None, gpus=(), poll_seconds=60, log=print,
                   watchdog=None):
    """Submit a job and wait for it to finish."""
    job.queue(lane=lane, hostname=hostname, gpus=list(gpus))
    return wait_for_job(job, poll_seconds=poll_seconds, log=log, watchdog=watchdog)


def wait_for_job(job, poll_seconds=60, log=print, watchdog=None):
    """Wait for an already-submitted job, reporting on status change and on a timer.

    `wait_for_done` is not used as-is because a class_2D runs for hours: spotting the
    known failure where the heartbeat stays alive while the job has effectively stopped
    needs the elapsed time. Submission and waiting are separate so that this can
    re-attach to a job another process submitted.

    Passing a watchdog calls watchdog(job, status, elapsed_seconds) on every poll. To
    abort the wait, the watchdog raises.
    """
    started = time.time()
    last_status = None
    last_report = 0.0
    while True:
        job.refresh()
        status = job.doc.get("status")
        elapsed = time.time() - started

        if status != last_status or elapsed - last_report >= poll_seconds:
            log(f"  {job.uid} {status} ({elapsed / 60:.1f} min)")
            last_status, last_report = status, elapsed

        if status in TERMINAL_STATUSES:
            break
        if watchdog:
            watchdog(job, status, elapsed)
        time.sleep(min(poll_seconds, 15))

    if status != "completed":
        raise SystemExit(f"{job.uid} ended as {status}")
    return job
