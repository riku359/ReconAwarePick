#!/usr/bin/env bash
# The CryoPPP catalogue: the spreadsheet that says which micrographs of each EMPIAR
# entry are the annotated 300. Every downloader after this one reads it.

. "$(dirname "$0")/_common.sh"

CATALOG="$DATA/cryoppp_tools/cryoppp"

banner "CryoPPP catalogue"
if [ -d "$CATALOG/.git" ]; then
  echo "  already present: $CATALOG"
else
  mkdir -p "$(dirname "$CATALOG")"
  git clone --quiet --depth 1 https://github.com/BioinfoMachineLearning/cryoppp.git "$CATALOG"
  echo "  -> $CATALOG"
fi
