"""The per-entry constants and the loop's arms, in one table each.

Everything that differs between EMPIAR entries lives in `ENTRIES`, so adding one is a
table row rather than a fork, and everything that differs between arms lives in `ARMS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import paths

# The two scales of a dataset config, named as docs/CONFIGURATION.md names them. Every
# round of the loop runs at `annot`; only the final evaluation runs at `full`.
SETTING_ANNOT = "annot"
SETTING_FULL = "full"

# How many micrographs `annot` holds.
SUBSET_MICROGRAPHS = 300


@dataclass(frozen=True)
class Entry:
    """One EMPIAR entry, and every constant that follows from it.

    diameter_px          the particle diameter this repository scores at. The GT-aligned
                         STAR carries centres only, so fine-tuning builds its boxes from
                         this.
    psize_A              pixel size, copied into every generated dataset config. 10345's
                         0.673 is CryoPPP's own value and is understated by 2x against
                         EMDB, so every resolution reported for that entry is half the
                         physical one; the pipeline follows CryoPPP throughout.
    box_size_pix         the extraction box. A different box changes every stack below it.
    fullset_micrographs  the full deposition's micrograph count, used as a floor: a Patch
                         CTF that finishes `completed` while silently dropping
                         micrographs is caught by comparing against this rather than
                         against whatever arrived.
    abinit_cap           where ab-initio reconstruction clamps its input. An arm above
                         the cap and an arm below it are not comparable, because the
                         particle-count axis is gone for one of them. None where no run
                         of this entry has ever reached a cap.
    gate                 what theta_0 scores on this entry's 300 annotated micrographs at
                         the loop's operating point (Sec. S2). Round 0 has to reproduce
                         it before anything downstream runs.
    """

    empiar: str
    diameter_px: int
    psize_A: float
    box_size_pix: int
    fullset_micrographs: int
    abinit_cap: Optional[int]
    gate: dict = field(default_factory=dict)

    @property
    def gt_star(self) -> Path:
        return paths.gt_star(self.empiar)

    @property
    def micrographs(self) -> Path:
        return paths.annotated_micrographs(self.empiar)

    @property
    def mask_dir(self) -> Path:
        return paths.mask_dir(self.empiar)


# MicrographCleaner's own training split -- unrelated to picker leakage -- puts 10081 and
# 10093 in distribution and 10345 and 10532 out of it. It changes nothing here (the masks
# are stored per entry either way) but it belongs with any claim made about the
# contamination stage on these entries.
ENTRIES = {
    "10081": Entry("10081", diameter_px=154, psize_A=1.3, box_size_pix=256,
                   fullset_micrographs=997, abinit_cap=219_900,
                   gate={"macro_P": 0.530, "macro_R": 0.919, "macro_F1": 0.655,
                         "n_pred_eval": 65386}),
    "10093": Entry("10093", diameter_px=172, psize_A=1.22, box_size_pix=256,
                   fullset_micrographs=1873, abinit_cap=207_000,
                   gate={"macro_P": 0.370, "macro_R": 0.737, "macro_F1": 0.491,
                         "n_pred_eval": 112226}),
    # The iterative 2D selection permanently rejects 88.5% of round 0's classes on this
    # entry: its reject threshold is absolute and does not adapt to an entry whose score
    # distribution sits higher. Teacher labels downstream of that are suspect, and a
    # round-over-round decline here may be the selector's score distribution rather than
    # the feedback loop.
    "10345": Entry("10345", diameter_px=149, psize_A=0.673, box_size_pix=300,
                   fullset_micrographs=1644, abinit_cap=None,
                   gate={"macro_P": 0.232, "macro_R": 0.947, "macro_F1": 0.354,
                         "n_pred_eval": 69744}),
    "10532": Entry("10532", diameter_px=174, psize_A=1.03, box_size_pix=256,
                   fullset_micrographs=1556, abinit_cap=176_700,
                   gate={"macro_P": 0.459, "macro_R": 0.580, "macro_F1": 0.498,
                         "n_pred_eval": 108456}),
}


# Where a round's training labels come from. `picks` is the loop proper: the particles
# that survived contamination removal and 2D class selection. `gt` replaces them with the
# CryoPPP annotations of the same micrographs, which is the perfect-teacher upper bound of
# Table 7 and not a feedback loop at all.
TEACHER_PICKS = "picks"
TEACHER_GT = "gt"


@dataclass(frozen=True)
class Arm:
    """One arm of the loop.

    finetune_mode        what `finetune.py` updates. The paper's is
                         `head_decoder_encoder_resnet`: every weight except resnet's
                         layer1, which stays frozen.
    source_prefix        the reconstruction source name each round is recorded under,
                         with the round number appended (fb_r0, fb_r1, ...).
    workspace_suffix     appended to the entry id to name the CryoSPARC workspace, so
                         two arms of one entry never share one.
    masked               whether picks landing on contamination are discarded.
    teacher              which labels the fine-tune trains on: TEACHER_PICKS (the
                         surviving picks) or TEACHER_GT (the CryoPPP annotations of the
                         same micrographs).
    in_paper             whether the paper reports this arm.
    """

    name: str
    finetune_mode: str
    source_prefix: str
    workspace_suffix: str
    masked: bool
    in_paper: bool
    note: str
    teacher: str = TEACHER_PICKS


# The paper's method is the default and the only arm it reports. The LoRA arms of the
# original study are not here: the paper fine-tunes all weights with resnet layer1
# frozen, LoRA was measured and dropped, and leaving a low-rank arm in as the default --
# which is how the original driver shipped -- meant that following the documentation
# reproduced something the paper does not describe.
ARMS = {
    "fb": Arm(
        name="fb",
        finetune_mode="head_decoder_encoder_resnet",
        source_prefix="fb_r",
        workspace_suffix="",
        masked=True,
        in_paper=True,
        note="the paper's method (Sec. 3.5): all weights, resnet layer1 frozen"),
    # NOT USED IN THE PAPER. Kept because it costs one table row and the existing
    # passthrough branch: it isolates the teacher labels from the contamination stage by
    # letting every pick through. Only two entries can answer anything with it -- the
    # mask removes 5.56% of picks on 10081 and 4.63% on 10532, against 0.13% on 10093,
    # while on 10345 the selector's collapse and the mask cannot be told apart.
    "fb_nomask": Arm(
        name="fb_nomask",
        finetune_mode="head_decoder_encoder_resnet",
        source_prefix="fbnm_r",
        workspace_suffix="_nomask",
        masked=False,
        in_paper=False,
        note="NOT IN THE PAPER: the same arm with the contamination stage skipped"),
    # The lower row of Table 7: one round with the teacher labels replaced by the CryoPPP
    # annotations of the same micrographs, everything else held fixed. It answers what a
    # perfect teacher would buy, so it is an upper bound rather than a feedback loop, and
    # it is run for round 0 only (`--rounds 0 --teacher gt`).
    #
    # REIMPLEMENTED FROM A WRITTEN PROCEDURE. The scripts that produced the published
    # numbers were never committed; this arm rebuilds what they are documented to have
    # done and has not been run end to end in this form.
    "fb_gt": Arm(
        name="fb_gt",
        finetune_mode="head_decoder_encoder_resnet",
        source_prefix="fbgt_r",
        workspace_suffix="_gt",
        masked=True,
        in_paper=True,
        note="the perfect-teacher upper bound (Table 7, lower row): the same arm with "
             "the CryoPPP annotations as the teacher",
        teacher=TEACHER_GT),
}

DEFAULT_ARM = "fb"

# The GT-teacher counterpart of an arm, so that `--teacher gt` never writes into the
# arm it is being read against: its rounds, checkpoints, reconstruction sources and
# CryoSPARC workspace are all separate.
GT_ARM_OF = {"fb": "fb_gt"}


def arm_for(name: str, teacher: str = None) -> "Arm":
    """The arm `--arm name --teacher teacher` selects; `teacher=None` means the arm's own.

    The teacher is not a free axis: an arm carries its own outputs, so swapping the
    labels means running the counterpart arm rather than overwriting this one's
    checkpoints.
    """
    arm = ARMS[name]
    if teacher is None or teacher == arm.teacher:
        return arm
    if teacher == TEACHER_GT and name in GT_ARM_OF:
        return ARMS[GT_ARM_OF[name]]
    raise ValueError(
        f"arm {name!r} trains on the {arm.teacher!r} teacher and has no {teacher!r} "
        f"counterpart; --teacher {teacher} is available for "
        f"{', '.join(sorted(GT_ARM_OF))}")


def loop_root(empiar: str, arm: str = DEFAULT_ARM) -> Path:
    """$RAPICK_WORK/loop/<id> for the paper's arm, /loop/<id>_<arm> for any other."""
    leaf = empiar if arm == DEFAULT_ARM else f"{empiar}_{arm}"
    return paths.work_root() / "loop" / leaf


def round_dir(empiar: str, n: int, arm: str = DEFAULT_ARM) -> Path:
    """$RAPICK_WORK/loop/<id>/round<n>: one round's state, labels and logs."""
    return loop_root(empiar, arm) / f"round{n}"


def model_path(empiar: str, n: int, arm: str = DEFAULT_ARM) -> Path:
    """The checkpoint round `n` picks with. Round 0 picks with theta_0 itself."""
    if n == 0:
        return paths.base_checkpoint()
    return loop_root(empiar, arm) / "models" / f"model_{n}.pth"


def fullset_dir(empiar: str, tag: str) -> Path:
    """Where one fullset evaluation of one checkpoint keeps its state and stars."""
    return paths.work_root() / "loop" / empiar / "fullset" / tag


def source_name(arm: str, n: int) -> str:
    """The reconstruction source name round `n` of `arm` is recorded under."""
    return f"{ARMS[arm].source_prefix}{n}"
