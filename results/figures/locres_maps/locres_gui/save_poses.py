# Freeze the current session: the camera, where every map sits, and which volume each
# panel is actually drawn from.
#
# Type this in the ChimeraX command line of the session you have been turning:
#     open results/figures/locres_maps/locres_gui/save_poses.py
#
# It writes $LOCRES_WORK/poses_<entry>.json, which render_locres_3d.py --poses reads
# instead of re-deriving the placement. Nothing is refitted at render time, so the
# figure shows the arrangement that was approved by hand -- which matters most where
# the fit used random restarts and would not repeat.
import json
import os

from chimerax.core.models import Model

P = json.load(open(os.environ["LOCRES_GUI_PARAMS"]))
ENTRY = P["entry"]
LABELS = [panel["label"] for panel in P["panels"]]
# A frozen state is the only record of a placement decided by hand, so never
# clobber one: a second freeze of the same entry goes to its own name unless the
# caller says otherwise, and an existing file is kept as .bak first.
WORK = os.environ.get("LOCRES_WORK", "/tmp/locres")
OUT = os.environ.get("LOCRES_POSES_OUT", "%s/poses_%s.json" % (WORK, ENTRY))
SIDECAR = "%s/files_%s.json" % (WORK, ENTRY)

# The session records what it drew each panel from: a mirrored panel comes from a
# volume ChimeraX derived and wrote out, not from the file the spec names.
try:
    with open(SIDECAR) as handle:
        files = json.load(handle)
except (IOError, ValueError):
    files = {panel["label"]: {"map": panel["masked"], "locres": panel["locres"]}
             for panel in P["panels"]}


def matrix_of(place):
    return [[float(v) for v in row] for row in place.matrix]


def label_of(model):
    """`10532_CryoSegNet__masked.mrc` -> `CryoSegNet`, and only for the drawn maps."""
    name = getattr(model, "name", "")
    if not name.startswith(ENTRY + "_") or "__masked" not in name:
        return None
    return name[len(ENTRY) + 1:].split("__masked")[0]


panels = {}
for model in session.models.list(type=Model):
    label = label_of(model)
    if label in LABELS and label not in panels:
        panels[label] = dict(files[label], matrix=matrix_of(model.position))

missing = [label for label in LABELS if label not in panels]
if missing:
    print("save_poses: no map found for " + ", ".join(missing))

state = {
    "entry": ENTRY,
    "reference": P["reference"],
    "palette": P["palette"],
    "camera": matrix_of(session.main_view.camera.position),
    "panels": panels,
}
if os.path.exists(OUT):
    backup = OUT + ".bak"
    index = 1
    while os.path.exists(backup):
        index += 1
        backup = "%s.bak%d" % (OUT, index)
    os.rename(OUT, backup)
    print("save_poses: kept the previous freeze as " + backup)
with open(OUT, "w") as handle:
    json.dump(state, handle, indent=1)
print("save_poses: wrote %s  (%d maps: %s)"
      % (OUT, len(panels), ", ".join(sorted(panels))))
