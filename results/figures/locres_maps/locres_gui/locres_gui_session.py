# ChimeraX script: open one entry's five maps in the figure's own frame, all in one
# session, so the alignment can be turned around by hand instead of trusted.
#
#   LOCRES_GUI_PARAMS=params_10532.json LOCRES_WORK=/tmp/locres \
#       ChimeraX --script locres_gui_session.py
#
# The placement is byte-for-byte the figure's: the reference is put on the principal
# axes of its own density, every other map starts from its own principal frame and is
# rigid-fit onto the reference with `fitmap`, and the row / panel turns of the render
# command are applied on top. What the figure never did is check the result, so this
# prints, per panel, the fit correlation and the angle its principal axes end up at
# relative to the reference. Zero means the frames agree.
#
# Each locres volume is moved with its own map, so `color sample` stays valid after
# the fit and the panels can be recoloured at any time -- the render script colours
# before it moves precisely because it does not do this.
import json
import os

import numpy as np
from chimerax.core.commands import run

P = json.load(open(os.environ["LOCRES_GUI_PARAMS"]))
PANELS = P["panels"]
FLAT = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759"]

# Envelope fitting: a gaussian-smoothed, coarsely-stepped copy is fitted with a global
# search and both hands are tried. The figure's own fit works on the raw masked density
# and cannot cross a mirror, which is why it settles tens of degrees out on the panels
# whose maps came out of refinement in the other hand.
FIT_SDEV = 6.0
FIT_STEP = 2
FIT_SEARCH = 12


def one(result):
    return result[0] if isinstance(result, (list, tuple)) else result


def matrix_command(rotation, centroid, model):
    rotation = np.asarray(rotation, dtype=float)
    translation = -rotation @ np.asarray(centroid, dtype=float)
    numbers = ",".join("%.6g" % v for row, shift in zip(rotation, translation)
                       for v in (*row, shift))
    return "view matrix models %s,%s" % (model, numbers)


def scene_axes(model, panel):
    """The panel's three principal axes as unit vectors in scene coordinates."""
    return np.array([model.position.transform_vector(panel[k])
                     for k in ("axis_mid", "axis_long", "axis_short")])


def angle_between(u, v):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)), -1.0, 1.0))))


def smoothed(volume):
    """A gaussian-smoothed, coarsely-stepped copy, for fitting the envelope."""
    copy = one(run(session, "volume gaussian #%s sDev %g"
                   % (volume.id_string, FIT_SDEV)))
    run(session, "volume #%s step %d" % (copy.id_string, FIT_STEP))
    run(session, "hide #%s models" % copy.id_string)
    run(session, "show #%s models" % volume.id_string)
    return copy


def mirrored_copy(volume):
    """A z-mirrored copy, the hand CryoSPARC's ab-initio may have picked instead."""
    copy = one(run(session, "volume flip #%s axis z" % volume.id_string))
    run(session, "show #%s models" % volume.id_string)
    return copy


def fit_envelope(volume, target):
    """Fit `volume`'s envelope into `target`'s and return the best correlation.

    A global search can come back with no cluster at all when no random start clears
    its threshold, so fall back to a local fit and then to a plain measurement rather
    than let one hard panel abort the session."""
    fits = run(session, "fitmap #%s inMap #%s search %d metric correlation"
               % (volume.id_string, target.id_string, FIT_SEARCH))
    if not fits:
        fits = run(session, "fitmap #%s inMap #%s metric correlation"
                   % (volume.id_string, target.id_string))
    if fits:
        return fits[0].correlation()
    correlation, _ = run(session, "measure correlation #%s inMap #%s"
                         % (volume.id_string, target.id_string))
    return correlation


# --- open -------------------------------------------------------------------
maps, locres = [], []
for panel in PANELS:
    maps.append(one(run(session, "open %s" % panel["masked"])))
    locres.append(one(run(session, "open %s" % panel["locres"])))

# --- hand, decided before anything is coloured or moved ---------------------
# A mirrored map has to be mirrored back before `color sample` runs, and its
# local-resolution volume with it: the two share a box, so the same flip keeps them
# in register.
MIRRORED = {}
MIRROR_DIR = os.path.dirname(PANELS[0]["masked"])
FILES = {panel["label"]: {"map": panel["masked"], "locres": panel["locres"]}
         for panel in PANELS}
if P.get("fit") == "envelope":
    reference_index = next((i for i, p in enumerate(PANELS)
                            if p["label"] == P["reference"]), 0)
    envelope_reference = smoothed(maps[reference_index])
    for index, panel in enumerate(PANELS):
        if index == reference_index:
            continue
        direct = smoothed(maps[index])
        flipped_map = mirrored_copy(maps[index])
        flipped = smoothed(flipped_map)
        as_is = fit_envelope(direct, envelope_reference)
        mirrored = fit_envelope(flipped, envelope_reference)
        MIRRORED[panel["label"]] = (mirrored > as_is, as_is, mirrored)
        if mirrored > as_is:
            flipped_locres = mirrored_copy(locres[index])
            run(session, "close #%s" % maps[index].id_string)
            run(session, "close #%s" % locres[index].id_string)
            maps[index] = flipped_map
            locres[index] = flipped_locres
            stem = "%s/%s_%s__" % (MIRROR_DIR, P["entry"], panel["label"])
            run(session, "save %smasked_flip.mrc models #%s"
                % (stem, flipped_map.id_string))
            run(session, "save %slocres_flip.mrc models #%s"
                % (stem, flipped_locres.id_string))
            FILES[panel["label"]] = {"map": stem + "masked_flip.mrc",
                                     "locres": stem + "locres_flip.mrc"}
        else:
            run(session, "close #%s" % flipped_map.id_string)
        run(session, "close #%s" % direct.id_string)
        run(session, "close #%s" % flipped.id_string)
    run(session, "close #%s" % envelope_reference.id_string)

for volume, other in zip(maps, locres):
    # No explicit level: the figure draws whatever ChimeraX picks on open.
    run(session, "show #%s models" % volume.id_string)
    run(session, "volume #%s style surface" % volume.id_string)
    run(session, "volume #%s style image hide" % other.id_string)

run(session, "set bgColor white")
run(session, "lighting soft")
run(session, "graphics silhouettes true width 1")

reference_index = next((i for i, p in enumerate(PANELS)
                        if p["label"] == P["reference"]), 0)
reference = maps[reference_index]

# --- colour, then place -----------------------------------------------------
for volume, other in zip(maps, locres):
    run(session, "color sample #%s map #%s palette %s"
        % (volume.id_string, other.id_string, P["palette"]))

run(session, "view initial")
run(session, matrix_command(PANELS[reference_index]["rotation"],
                            PANELS[reference_index]["centroid"],
                            "#" + reference.id_string))
smoothed_reference = None
if P.get("fit") == "envelope":
    smoothed_reference = smoothed(reference)
    smoothed_reference.position = reference.position

report = []
for index, (panel, volume) in enumerate(zip(PANELS, maps)):
    if index == reference_index:
        report.append((panel["label"], None))
        continue
    run(session, matrix_command(panel["rotation"], panel["centroid"],
                                "#" + volume.id_string))
    if P.get("fit") == "envelope":
        # Fit the envelopes, then carry the pose over to the map that is drawn.
        envelope = smoothed(volume)
        envelope.position = volume.position
        correlation = fit_envelope(envelope, smoothed_reference)
        volume.position = envelope.position
        run(session, "close #%s" % envelope.id_string)
        run(session, "show #%s models" % volume.id_string)
    else:
        fits = run(session, "fitmap #%s inMap #%s" % (volume.id_string,
                                                      reference.id_string))
        correlation = fits[0].correlation() if fits else float("nan")
    report.append((panel["label"], correlation))

# The render script applies a panel turn as a camera turn in that panel's own
# session, which comes to the same thing as turning that map relative to the
# reference. Here the maps share a session, so it has to be the map that turns.
for panel, volume in zip(PANELS, maps):
    command = P["panel_turns"].get(panel["label"])
    if command:
        axis_angle = command.split()[1:]
        run(session, "turn %s center #%s coordinateSystem scene models #%s"
            % (" ".join(axis_angle), reference.id_string, volume.id_string))

# Each locres volume follows its map, so the colouring can be redone later.
for volume, other in zip(maps, locres):
    other.position = volume.position

if P["turn"]:
    run(session, P["turn"])                      # row turn: the whole scene
run(session, "view #%s" % reference.id_string)
run(session, "view name figure")

# --- what the figure never checked ------------------------------------------
reference_axes = scene_axes(reference, PANELS[reference_index])
print("")
print("EMPIAR-%s   reference %s   palette %s"
      % (P["entry"], P["reference"], P["palette"]))
print("  %-16s %6s  %8s %8s %8s  %8s"
      % ("panel", "corr", "long", "mid", "short", "frame"))
for (label, correlation), panel, volume in zip(report, PANELS, maps):
    if correlation is None:
        print("  %-16s %6s  %8s %8s %8s  %8s"
              % (label, "--", "0.0", "0.0", "0.0", "0.0"))
        continue
    axes = scene_axes(volume, panel)
    per_axis = [angle_between(axes[i], reference_axes[i]) for i in range(3)]
    relative = axes.T @ reference_axes            # both are orthonormal row sets
    frame = float(np.degrees(np.arccos(
        np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
    hand = MIRRORED.get(label)
    print("  %-16s %6.3f  %8.1f %8.1f %8.1f  %8.1f  %s"
          % (label, correlation, per_axis[1], per_axis[0], per_axis[2], frame,
             ("mirrored (%.3f vs %.3f)" % (hand[2], hand[1])) if hand and hand[0] else ""))
print("  angles in degrees, principal axes after the fit, against the reference's.")
print("")

# --- handles for looking at it ----------------------------------------------
ids = [v.id_string for v in maps]
extent = reference.bounds().xyz_max[0] - reference.bounds().xyz_min[0]
spacing = extent * 1.15

run(session, "alias stack " + " ; ".join(
    "move x %.4g models #%s coordinateSystem scene" % (-spacing * i, ids[i])
    for i in range(len(ids))))
run(session, "alias sbs " + " ; ".join(
    "move x %.4g models #%s coordinateSystem scene" % (spacing * i, ids[i])
    for i in range(len(ids))))
run(session, "alias flat " + " ; ".join(
    "color #%s %s" % (ids[i], FLAT[i % len(FLAT)]) for i in range(len(ids))))
run(session, "alias locres " + " ; ".join(
    "color sample #%s map #%s palette %s" % (ids[i], locres[i].id_string, P["palette"])
    for i in range(len(ids))))
run(session, "alias ghost transparency #%s 60 surfaces" % ",#".join(ids))
run(session, "alias solid transparency #%s 0 surfaces" % ",#".join(ids))
run(session, "alias pall show #%s models" % ",#".join(ids))
for i, panel in enumerate(PANELS):
    run(session, "alias p%d hide #%s models ; show #%s models"
        % (i + 1, ",#".join(ids), ids[i]))
run(session, "alias figview view figure")

with open("%s/files_%s.json" % (os.environ.get("LOCRES_WORK", "/tmp/locres"),
                                P["entry"]), "w") as handle:
    json.dump(FILES, handle, indent=1)

print("commands:  flat / locres    colour by method, or by local resolution")
print("           ghost / solid    transparency, for seeing the envelopes overlap")
print("           sbs / stack      spread the five apart, or superimpose them")
print("           p1..p%d / pall    show one panel, or all" % len(PANELS))
print("           figview          back to the figure's camera")
print("panels:    " + "  ".join("p%d=%s" % (i + 1, p["label"])
                                for i, p in enumerate(PANELS)))
