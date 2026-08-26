"""Contamination masking (Sec. 3.3, Sec. S3).

MicrographCleaner's pretrained model predicts contamination as a per-pixel probability
map, and a candidate is discarded when the map reaches 0.5 at the candidate's centre.
The per-window predictions are assembled by triangular blending (triangular_mask.py)
rather than by the released uniform averaging plus `fixJumpInBorders` seam repair.
"""
