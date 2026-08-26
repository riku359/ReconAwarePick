"""2D class selection (Sec. 3.4, Sec. S4).

CryoSift's pretrained CNN scores every CryoSPARC 2D class average from 1.0 (a clean
particle class) to 5.0 (a non-particle class), and the classes are then selected either
at a single cutoff (purify_class2d.py) or through the paper's iterative workflow
(iterate_class2d.py).
"""
