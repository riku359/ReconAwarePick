# Fig. 1: the pipeline

Hand-drawn: `build_pipeline_fig.py` draws the whole deck with python-pptx, and the box
labels live in the script rather than in the `.pptx`, so the deck is an output and not a
source.

Runs standalone once the photographic assets are in place. No CryoSPARC, no ChimeraX.

## Assets

Not committed: they are photographic, and the repository commits code and numbers. Point
`--assets` at a directory holding

```
q_raw.jpg  q_bbox.jpg  q_masked.jpg     the three micrograph thumbnails
patches/patch_<nnn>.jpg                 raw particle crops
plain/cls_k0.png ... cls_d1.png         class averages with no burnt-in frame
map3d_10081_gt_transparent.png          the 3D volume, transparent background
```

The class averages are `plain` because the versions CryoSPARC renders carry a burnt-in
green or red frame; the script draws the frame itself, in the colour of the class's fate.

## Build

```bash
python build_pipeline_fig.py --assets <dir> --out /tmp/pipeline_overview.pptx
soffice --headless --convert-to pdf --outdir /tmp /tmp/pipeline_overview.pptx
```

LibreOffice writes the full 16:9 slide, so the last step trims it to the drawing.

**Crop by setting the page boxes, never by re-rendering through `gs -sDEVICE=pdfwrite`.**
That route re-encodes the fonts and breaks the ligatures in the extracted text.

```bash
python - <<'PY'
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
BOX = (0.06, 95.75, 960.0, 539.26)   # ink bbox + 7 pt on each side, clamped to the page
r = PdfReader("/tmp/pipeline_overview.pdf"); w = PdfWriter()
pg = r.pages[0]; pg.mediabox = pg.cropbox = RectangleObject(BOX); w.add_page(pg)
w.write(open("/tmp/pipeline_overview_cropped.pdf", "wb"))
PY
```

`BOX` is the frame the figure in the paper uses. Recompute it only if the layout changes:
`gs -q -dBATCH -dNOPAUSE -sDEVICE=bbox <pdf>` gives the ink bounding box, and the frame is
that box widened by 7 pt on each side.

## Reading the layout

Two blocks. The top block is one round of the feedback loop, with no reconstruction in it
because the loop does not run one. The bottom block picks the full micrograph set with the
checkpoint the loop delivers and takes it through the same contamination mask and 2D class
selection to reconstruction.

The top block is a serpentine: row 1 runs left to right, turns at the right edge, row 2
runs right to left. Putting fine-tune directly under CryoTransformer is what lets the
returned checkpoint be one vertical dashed line rather than a path around the block.

Font sizes are chosen for the page, not for the screen. The paper's text width is
6.875 in and the figure is a 13.333 in wide PDF placed at `\linewidth`, so it lands at
about 0.52 scale; matching the 10 pt body text needs about 19 pt in the deck.

The symbols match Sec. 3 of the paper: M, C_n, M_i, C'_n, S_n, T_n, theta_n, theta_{n+1}.

## Not ported

Two earlier versions of this figure were left behind: the first draft, and a
single-block version that nothing in the manuscript references.
