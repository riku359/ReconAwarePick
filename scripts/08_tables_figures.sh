#!/usr/bin/env bash
# Rebuild the tables and the figures that need nothing but this repository.
#
# Most of the paper's figures are drawn over micrographs, class averages or density
# maps, and none of those is committed: they are large binaries, and this repository
# commits code and numbers. Those figures take an --assets directory instead, and
# results/figures/README.md says what each one wants. What runs here is the part
# that reads only results/tables/.
#
#   bash scripts/08_tables_figures.sh              figures that stand alone
#   bash scripts/08_tables_figures.sh --tables     re-derive the 2D tables instead
#   bash scripts/08_tables_figures.sh --list       what is here and what it needs
#
# Figures land in $RAPICK_FIGURES_OUT, default $RAPICK_WORK/figures.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="figures"
while [ $# -gt 0 ]; do
  case "$1" in
    --tables)  MODE="tables"; shift ;;
    --list)    MODE="list"; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [ "$MODE" = "list" ]; then
  sed -n '/^## Index/,/^\[`lib\//p' "$REPO/results/figures/README.md"
  exit 0
fi

require_roots
OUT="${RAPICK_FIGURES_OUT:-$WORK/figures}"
mkdir -p "$OUT"
PY="$(venv_python figures)"
FIG="$REPO/results/figures"

if [ "$MODE" = "tables" ]; then
  # The 2D numbers are the only ones this repository can re-derive without a
  # CryoSPARC instance: everything resolution-shaped comes out of a run.
  banner "2D detection scores (Table S2)"
  EV="$(venv_python cryotransformer)"
  "$EV" "$REPO/src/rapick/eval/calc_common_2d_metrics.py" --batch --markdown \
      --out-json "$OUT/detection_2d.json"
  echo
  echo "Written: $OUT/detection_2d.json"
  echo "Compare against results/tables/detection_2d.json."
  echo
  echo "Table 3 needs the baseline and mask conditions scored on the annotated set:"
  echo "  python3 results/analysis/ablation_2d_metrics.py --entries ${ENTRIES_ALL[*]}"
  echo "Every other table is read out of a run's metrics.json; see docs/REPRODUCE.md."
  exit 0
fi

banner "Figures that read only results/tables/"

# Fig. 4. Deliberately reads the committed tables rather than a run, so the figure
# and the table it plots cannot drift apart.
"$PY" "$FIG/f1_vs_resolution/build_f1_vs_resolution.py" --out "$OUT/f1_vs_resolution.pdf"
echo "  Fig. 4  -> $OUT/f1_vs_resolution.pdf"

# Fig. 5.
"$PY" "$FIG/cryosift_scores/build_cryosift_scores.py" \
    --csv "$REPO/results/tables/cryosift_class_scores.csv" \
    --out "$OUT/cryosift_scores.pdf"
echo "  Fig. 5  -> $OUT/cryosift_scores.pdf"

# The figure form of Table 6. The manuscript carries it in its candidate
# supplement, so it has no figure number.
"$PY" "$FIG/loop_rounds/build_loop_rounds.py" --out "$OUT/loop_rounds.pdf"
echo "  loop rounds -> $OUT/loop_rounds.pdf"

echo
echo "Not built here, because each needs an asset directory or an external tool:"
echo "  Fig. 1, 2, 6      photographic assets, python-pptx and LibreOffice"
echo "  Fig. 3            ChimeraX and the refinement volumes"
echo "  Fig. S2, S3       one micrograph and its two masks (--assets)"
echo "  Fig. S5, S6, S7   panels fetched from a live CryoSPARC instance"
echo "Each directory's README under results/figures/ has the command."
