"""Shared types for job modules.

A job module's `run` returns a JobResult. `JobResult.outputs` maps the logical
output names (as used by downstream steps) to the actual CryoSPARC output port
names, so a downstream step wires inputs without hard-coding port strings.

Job modules build their `connections` dict explicitly (rather than via a generic
helper) so that each stage's wiring is readable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JobResult:
    name: str                  # research stage name, e.g. "ctf"
    job_uid: str               # CryoSPARC job UID, e.g. "J11"
    job_type: str              # e.g. "patch_ctf_estimation_multi"
    status: str                # e.g. "completed"
    outputs: dict = field(default_factory=dict)   # logical name -> output port name


@dataclass
class JobSpec:
    """Resolved-from-profile description of a step (convenience for tests/inspection)."""
    step: str
    job_type: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    fixed_params: dict = field(default_factory=dict)
