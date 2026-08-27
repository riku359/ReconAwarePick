#!/usr/bin/env python3
"""Preparing a CryoSPARC refinement volume so ChimeraX draws the molecule.

Three things have to be got right, and all three were wrong in the first attempt at
this figure. They are separated here so the renderer and any diagnostic script use
the same code.

1. MASK THE MAP. The sharpened volume `J*_volume_map_sharp.mrc` is sharpened over the
   whole box, so outside the molecule it is high-frequency noise at a level comparable
   to the molecular surface. Contour it directly and the surface is a solid ball of
   noise filling the refinement sphere. Multiplying by the refinement mask
   `J*_mask_refine.mrc` removes it.

2. CONTOUR BY ENCLOSED VOLUME, NOT BY A FIXED PERCENTILE. A percentile of all voxels
   is meaningless across maps because the molecule occupies a different fraction of
   each box: measured over the four entries here it runs from 3.4% to 9.5%. The mask
   states that fraction per map, so the level is chosen to enclose a fixed multiple of
   the mask's own volume. 0.6 is about right; the CryoSPARC mask is dilated, so
   enclosing all of it swallows the molecule in its own envelope.

3. ORIENT BY THE DENSITY, NOT BY `view orient`. `view orient` picks a standard axis
   frame with no knowledge of the molecule, and on these entries it happened to look
   straight down the long axis of EMPIAR-10532, drawing an elongated trimer as a round
   ball. Orienting on the principal axes of the thresholded density instead puts the
   longest extent up the screen and views down the shortest, which shows the shape and
   also puts every panel of one molecule in the same frame without any map-to-map
   fitting.
"""

import numpy as np

try:
    import mrcfile
except ImportError:  # only the pure-numpy helpers are importable without it
    mrcfile = None

# Fraction of the refinement mask's volume the isosurface should enclose. The mask is
# a dilated envelope, so 1.0 draws the envelope rather than the molecule.
MASK_VOLUME_FRACTION = 0.6


def read(path):
    with mrcfile.open(str(path), permissive=True) as handle:
        return (np.asarray(handle.data, dtype=np.float32),
                float(handle.voxel_size.x),
                np.array([handle.header.origin.x, handle.header.origin.y,
                          handle.header.origin.z], dtype=np.float64))


def unsharpened(map_path):
    """The refinement's own map rather than its sharpened copy, when it exists.

    Sharpening amplifies high frequencies over the whole box, which leaves the
    isosurface crenellated: neighbouring vertices then sample quite different local
    resolutions and the panel is covered in speckle. The unsharpened volume is
    band-limited to the refinement's resolution and gives a surface whose colour can
    be read.
    """
    import os
    plain = str(map_path).replace("_volume_map_sharp.mrc", "_volume_map.mrc")
    return plain if os.path.exists(plain) else str(map_path)


def masked_density(map_path, mask_path):
    """The map with everything outside the refinement mask removed."""
    density, voxel, origin = read(map_path)
    mask, _, _ = read(mask_path)
    if density.shape != mask.shape:
        raise ValueError(f"shape mismatch: {density.shape} vs {mask.shape}")
    return density * mask, mask, voxel, origin


def contour_level(density, mask, fraction=MASK_VOLUME_FRACTION):
    """The value enclosing `fraction` of the mask's voxel count."""
    target = int((mask > 0.5).sum() * fraction)
    descending = np.sort(density.ravel())[::-1]
    return float(descending[min(max(target, 1), descending.size - 1)])


def principal_frame(density, level, voxel, origin):
    """Rotation and centroid that put the molecule's long axis up the screen.

    Returns (rotation, centroid) in scene coordinates (Angstrom). The rotation maps
    model coordinates to a frame where the largest principal axis is screen-y, the
    second is screen-x and the smallest is screen-z, so the camera looks down the
    shortest extent and the projected silhouette is as large as it can be.

    Sign flips are the one ambiguity principal axes leave. They are resolved by the
    skewness of the density along each axis, which is a property of the molecule, so
    two maps of the same molecule land in the same frame rather than in one of four.
    """
    occupied = np.argwhere(density >= level)      # (n, 3) as (z, y, x) index
    if len(occupied) < 10:
        raise ValueError("almost nothing above the contour level")
    weights = density[density >= level].astype(np.float64)

    # mrcfile indexes (z, y, x); scene coordinates are (x, y, z).
    points = occupied[:, ::-1].astype(np.float64) * voxel + origin
    centroid = np.average(points, axis=0, weights=weights)
    centred = points - centroid

    covariance = (centred * weights[:, None]).T @ centred / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]         # longest extent first
    axes = eigenvectors[:, order].T               # rows: long, mid, short

    for i, axis in enumerate(axes):
        projected = centred @ axis
        skew = np.average(projected ** 3, weights=weights)
        if skew < 0:
            axes[i] = -axis

    # Screen x = mid axis, y = long axis, z = short axis, kept right-handed.
    rotation = np.stack([axes[1], axes[0], axes[2]])
    if np.linalg.det(rotation) < 0:
        rotation[2] = -rotation[2]
    return rotation, centroid


def view_matrix_command(rotation, centroid, model="#1"):
    """ChimeraX `view matrix models` string placing the model in that frame."""
    translation = -rotation @ centroid
    numbers = ",".join(
        f"{value:.6g}"
        for row, shift in zip(rotation, translation)
        for value in (*row, shift)
    )
    return f"view matrix models {model},{numbers}"
