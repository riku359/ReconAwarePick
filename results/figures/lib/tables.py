"""Read the paper's numbers out of `results/tables/`.

Two figures plot values that a table also prints. They used to hold those values as
literals transcribed from the table, so that the figure and the table could not drift
apart. `results/tables/` now holds the same numbers together with their provenance, so
the literals are gone and both read the one file: the anti-drift property is kept and
the duplication is not.

Every table file is one JSON object with the same envelope:

    {"table": "tab:main_results",      the LaTeX label in the manuscript
     "paper_number": "Table 2",
     "caption": ..., "units": ..., "metric": ...,
     "note": ...,                      what a reader has to know to use the numbers
     "provenance": {...},              where each value came from
     "values": {...},                  the numbers the table prints
     "not_in_paper": {...}}            numbers the notes keep that the table does not

`values` is keyed by the release condition names of `docs/PAPER_TO_CODE.md`
(`baseline`, `mask`, `select`, `both`, `fb`, and the three other pickers), then by
EMPIAR entry. A missing file or a missing key raises with the path and the key in the
message, rather than plotting a hole.
"""
from __future__ import annotations

import json
from pathlib import Path

from figure_paths import TABLES_ROOT


def table_path(name: str) -> Path:
    """`results/tables/<name>.json`."""
    return TABLES_ROOT / f"{name}.json"


def load(name: str) -> dict:
    """One table file, as the whole envelope."""
    path = table_path(name)
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. It carries the numbers this figure plots; see "
            f"results/tables/README.md for what writes it.")
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}")


def _walk(name: str, node, keys, prefix: str):
    """Descend by successive keys, naming the level that is missing."""
    walked = prefix
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            available = sorted(node) if isinstance(node, dict) else "(not a mapping)"
            raise SystemExit(
                f"{table_path(name)}: no `{walked}[{key!r}]`; that level holds {available}")
        walked += f"[{key!r}]"
        node = node[key]
    return node


def values(name: str) -> dict:
    """The `values` block of one table file: what the table itself prints."""
    return _walk(name, load(name), ("values",), "")


def cell(name: str, *keys):
    """Walk `values` by successive keys.

        cell("main_results", "fb", "10081")  ->  {"published": 4.12, ...}
    """
    return _walk(name, values(name), keys, "values")


def extra(name: str, *keys):
    """Walk `not_in_paper`, the block holding numbers the table stopped printing."""
    return _walk(name, load(name), ("not_in_paper",) + keys, "")


def number(name: str, *keys, field=None) -> float:
    """One number from a table cell.

    A cell may be a bare number or a dict carrying the value under one of several field
    names along with its provenance; both are accepted so a figure does not have to know
    which. `field` is one name or a sequence of names to try in order.
    """
    node = cell(name, *keys)
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        wanted = [field] if isinstance(field, str) else list(field or ())
        for candidate in wanted + ["published", "value", "raw"]:
            if candidate in node and isinstance(node[candidate], (int, float)):
                return float(node[candidate])
        raise SystemExit(
            f"{table_path(name)}: cell {list(keys)} holds {sorted(node)}, none of which "
            f"is the number this figure wants ({', '.join(wanted) or 'a value'})")
    raise SystemExit(f"{table_path(name)}: cell {list(keys)} is a {type(node).__name__}")
