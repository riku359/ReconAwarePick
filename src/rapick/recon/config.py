"""Load and merge the four config layers into one resolved runtime config.

    profile   (configs/cryosparc_v47.yaml)     -> job-type / port contract
  + condition (configs/conditions/<name>.yaml) -> seeds, 2D params, pipeline flags
  + dataset   (configs/datasets/empiar_<id>.yaml) -> micrographs, optics, per-condition picks
  + .env                                       -> credentials + machine paths

`${VAR}` in any string is expanded from the merged env (.env layered under the
process environment). Secrets never live in a tracked file.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def load_env(path: str | Path) -> dict:
    """Parse a KEY=VALUE .env file, then layer the process environment on top
    (so an exported var wins over the file). Missing file is fine (env only)."""
    values: dict[str, str] = {}
    path = Path(path)
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def _expand(value: Any, env: dict) -> Any:
    """Recursively expand ${VAR} in strings against `env`."""
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            name = m.group(1)
            if name not in env:
                raise KeyError(f"env var {name!r} is not set (referenced in config)")
            return env[name]
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
    return value


def _load_yaml(path: str | Path, env: dict) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return _expand(yaml.safe_load(path.read_text()) or {}, env)


# --- connection ---------------------------------------------------------
@dataclass
class ConnectionConfig:
    host: str
    port: int
    license: str
    email: str
    password: str

    @classmethod
    def from_env(cls, env: dict) -> "ConnectionConfig":
        required = ["CRYOSPARC_LICENSE_ID", "CRYOSPARC_EMAIL", "CRYOSPARC_PASSWORD"]
        missing = [k for k in required if not env.get(k)]
        if missing:
            raise ValueError(f"missing in env: {', '.join(missing)}")
        return cls(
            host=env.get("CRYOSPARC_HOST", "localhost"),
            # CryoSPARC's own default base port; override with CRYOSPARC_PORT to
            # match your master's CRYOSPARC_BASE_PORT in cryosparc_master/config.sh.
            port=int(env.get("CRYOSPARC_PORT", "39000")),
            license=env["CRYOSPARC_LICENSE_ID"],
            email=env["CRYOSPARC_EMAIL"],
            password=env["CRYOSPARC_PASSWORD"],
        )


# --- profile (job-type / port contract) ---------------------------------
@dataclass
class Profile:
    jobs: dict

    def _entry(self, step: str) -> dict:
        try:
            return self.jobs[step]
        except KeyError:
            raise KeyError(f"unknown step {step!r}; profile has {sorted(self.jobs)}")

    def type(self, step: str) -> str:
        return self._entry(step)["type"]

    def inputs(self, step: str) -> dict:
        return self._entry(step).get("inputs", {})

    def outputs(self, step: str) -> dict:
        return self._entry(step).get("outputs", {})

    def fixed_params(self, step: str) -> dict:
        return self._entry(step).get("fixed_params", {})

    def seed_param(self, step: str) -> Optional[str]:
        """Single param that receives the trial seed (e.g. class2d, refine), or None."""
        return self._entry(step).get("seed_param")

    def seed_params(self, step: str) -> list:
        """Multiple params that receive the trial seed (e.g. abinit)."""
        return self._entry(step).get("seed_params", [])

    @classmethod
    def load(cls, path: str | Path, env: dict) -> "Profile":
        return cls(jobs=_load_yaml(path, env).get("jobs", {}))


# --- dataset ------------------------------------------------------------
@dataclass
class SourceConfig:
    name: str
    star: str
    import_params: dict           # extra params for Import Particles (e.g. query_cut_prefix)
    y_flip: bool = True           # flip Y (ny - Y) before import; GT-aligned stars need it


@dataclass
class SettingConfig:
    """One scale of one dataset: `annot` (the 300 CryoPPP-annotated micrographs) or
    `full` (the whole deposition). Each scale has its own micrographs, its own
    expected count, and its own per-condition picks."""
    name: str
    micrograph_glob: str
    expected_micrograph_count: Optional[int]
    sources: dict                    # condition name -> SourceConfig


@dataclass
class DatasetConfig:
    name: str
    empiar_id: str
    optics: dict
    box_size_pix: int
    settings: dict                   # setting name -> SettingConfig

    def setting(self, setting: str) -> SettingConfig:
        try:
            return self.settings[setting]
        except KeyError:
            raise KeyError(f"dataset {self.name} has no {setting!r} section; it defines "
                           f"{sorted(self.settings)}")

    def micrograph_glob(self, setting: str) -> str:
        return self.setting(setting).micrograph_glob

    def expected_count(self, setting: str) -> Optional[int]:
        """Minimum number of healthy micrographs required for a setting, or None if the
        dataset does not declare one (then only the broken-file check applies)."""
        return self.setting(setting).expected_micrograph_count

    def sources(self, setting: str) -> dict:
        """{condition name -> SourceConfig} declared for this scale."""
        return self.setting(setting).sources

    def source(self, setting: str, name: str) -> SourceConfig:
        sources = self.sources(setting)
        try:
            return sources[name]
        except KeyError:
            raise KeyError(
                f"{self.name}/{setting} declares no picks for condition {name!r}; it has "
                f"{sorted(sources)}. Conditions whose chain starts at an existing "
                f"select_2D job (select / both / cryosegnet_both) carry no STAR of their "
                f"own -- reconstruct them from their parent condition's 2D selection.")

    def import_params(self, setting: str) -> dict:
        """Params for the Import Micrographs job (blob_paths + optics)."""
        return {"blob_paths": self.micrograph_glob(setting), **self.optics}

    @classmethod
    def load(cls, path: str | Path, env: dict) -> "DatasetConfig":
        d = _load_yaml(path, env)
        settings = {}
        for name, block in (d.get("settings") or {}).items():
            settings[name] = SettingConfig(
                name=name,
                micrograph_glob=block["micrographs"],
                expected_micrograph_count=block.get("expected_micrograph_count"),
                sources={
                    src: SourceConfig(name=src, star=s["star"],
                                      import_params=s.get("import_params") or {},
                                      y_flip=bool(s.get("y_flip", True)))
                    for src, s in (block.get("sources") or {}).items()
                },
            )
        return cls(
            name=d["name"],
            empiar_id=str(d["empiar_id"]),
            optics=d.get("optics", {}),
            box_size_pix=int(d.get("extraction", {}).get("box_size_pix", 256)),
            settings=settings,
        )


# --- condition ----------------------------------------------------------
@dataclass
class ConditionConfig:
    """One condition of the paper (baseline / mask / select / both / fb / ...)."""
    name: str
    seeds: list
    class2d_params: dict
    select2d_enabled: bool
    choose_best_by: str
    local_res_enabled: bool
    artifacts: dict

    @classmethod
    def load(cls, path: str | Path, env: dict) -> "ConditionConfig":
        d = _load_yaml(path, env)
        pipe = d.get("pipeline", {})
        recon = pipe.get("reconstruction", {})
        return cls(
            name=d.get("name", "condition"),
            # Documentation only: --seeds must be passed explicitly on every run, so a
            # single-seed run is always a deliberate choice and never a silent default.
            seeds=recon.get("seeds", [0]),
            class2d_params=pipe.get("class2d", {}),
            select2d_enabled=bool(pipe.get("select2d", {}).get("enabled", False)),
            choose_best_by=recon.get("choose_best_by", "res_gsfsc_0143"),
            local_res_enabled=bool(pipe.get("local_resolution", {}).get("enabled", True)),
            artifacts=d.get("artifacts", {}),
        )


# --- bundle -------------------------------------------------------------
@dataclass
class ResolvedConfig:
    connection: ConnectionConfig
    profile: Profile
    condition: ConditionConfig
    dataset: DatasetConfig
    project_uid: str      # CRYOSPARC_PROJECT: the project uid jobs are created in
    worker: str           # CRYOSPARC_WORKER: the worker lane jobs are queued to
    work_root: str        # RAPICK_WORK: where manifest.json / metrics.json land


def resolve(env_path, profile_path, condition_path, dataset_path) -> ResolvedConfig:
    """Load all layers and assemble the runtime config."""
    env = load_env(env_path)
    return ResolvedConfig(
        connection=ConnectionConfig.from_env(env),
        profile=Profile.load(profile_path, env),
        condition=ConditionConfig.load(condition_path, env),
        dataset=DatasetConfig.load(dataset_path, env),
        project_uid=env.get("CRYOSPARC_PROJECT", ""),
        worker=env.get("CRYOSPARC_WORKER", ""),
        work_root=env.get("RAPICK_WORK", ""),
    )
