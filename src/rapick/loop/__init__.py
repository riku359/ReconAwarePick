"""rapick.loop: the reconstruction-aware feedback loop (paper Sec. 3.5).

Each round picks with the current checkpoint, discards the picks that land on
contamination, selects the survivors by 2D class, takes the surviving particles on 50
micrographs as pseudo-labels, and fine-tunes theta_0 -- the base checkpoint, not the one
that just picked -- on them:

    theta_{n+1} = FineTune(theta_0; S_n)                                    (Eq. 1)

Three rounds are run and the paper reports round 1. Fine-tuning updates all weights with
resnet layer1 frozen; it is not LoRA. See README.md in this package.
"""

__version__ = "0.1.0"
