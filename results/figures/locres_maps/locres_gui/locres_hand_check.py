# Does each panel's map need the opposite hand? CryoSPARC ab-initio picks a hand per
# run, and `fitmap` cannot cross a mirror, so a wrong-hand map settles at whatever
# rotation fits least badly and the render silently keeps it. This fits every panel
# twice -- as it is, and z-mirrored -- and prints both correlations.
import json
import os

import numpy as np
from chimerax.core.commands import run

P = json.load(open(os.environ["LOCRES_GUI_PARAMS"]))
PANELS = P["panels"]


def one(result):
    return result[0] if isinstance(result, (list, tuple)) else result


def matrix_command(rotation, centroid, model):
    rotation = np.asarray(rotation, dtype=float)
    translation = -rotation @ np.asarray(centroid, dtype=float)
    numbers = ",".join("%.6g" % v for row, shift in zip(rotation, translation)
                       for v in (*row, shift))
    return "view matrix models %s,%s" % (model, numbers)


def correlation_of(fits):
    return fits[0].correlation() if fits else float("nan")


reference_index = next((i for i, p in enumerate(PANELS)
                        if p["label"] == P["reference"]), 0)
reference = one(run(session, "open %s" % PANELS[reference_index]["masked"]))
run(session, matrix_command(PANELS[reference_index]["rotation"],
                            PANELS[reference_index]["centroid"],
                            "#" + reference.id_string))

print("")
print("EMPIAR-%s   hand check against %s" % (P["entry"], P["reference"]))
print("  %-16s %8s %8s   %s" % ("panel", "as-is", "mirrored", "verdict"))
for index, panel in enumerate(PANELS):
    if index == reference_index:
        continue
    direct = one(run(session, "open %s" % panel["masked"]))
    run(session, matrix_command(panel["rotation"], panel["centroid"],
                                "#" + direct.id_string))
    as_is = correlation_of(run(session, "fitmap #%s inMap #%s"
                               % (direct.id_string, reference.id_string)))

    flipped = one(run(session, "volume flip #%s axis z" % direct.id_string))
    run(session, matrix_command(panel["rotation"], panel["centroid"],
                                "#" + flipped.id_string))
    mirrored = correlation_of(run(session, "fitmap #%s inMap #%s"
                                  % (flipped.id_string, reference.id_string)))

    verdict = "MIRRORED" if mirrored > as_is + 0.02 else ""
    print("  %-16s %8.3f %8.3f   %s" % (panel["label"], as_is, mirrored, verdict))
    run(session, "close #%s" % direct.id_string)
    run(session, "close #%s" % flipped.id_string)
print("")
run(session, "exit")
