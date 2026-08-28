#!/usr/bin/env bash
# Recover anything that failed, then check what landed.
#
# A download can fail in ways an existence check does not catch: two workers appending
# to one .part file, or EBI's S3 endpoint returning an XML error body that lands inside
# the .mrc with a plausible size and only fails hours later at Patch CTF. And
# import_particles dies on the first missing micrograph, while a *.mrc glob happily
# imports a partial download, so the count is worth confirming here rather than three
# hours into a run.

. "$(dirname "$0")/_common.sh"

banner "Recovering anything that failed"
# --ids is not optional here. Left off, the recovery treats every one of CryoPPP's 34
# entries as in scope, finds the 33 that were never fetched entirely missing, and
# downloads a per-entry archive for each -- hundreds of GB, against an $RAPICK_ENTRIES
# that asked for one. This downloader wants one comma-separated list where the others
# take repeated ids ($ENTRIES_CSV, from _common.sh).
# Not fatal: there may simply be nothing to recover.
if ! run_dl "$DL/recover_failed_mrc_from_targz.py" --data-root "$DATA" \
        --ids "$ENTRIES_CSV" --max-retries 5; then
  echo "  (nothing to recover, or recovery reported failures; see the log under cryoppp_tools/)"
fi

banner "Verifying micrograph integrity"
# `subset` is the verifier's name for $RAPICK_DATA/cryoppp; `fullset` for
# cryoppp_fullset. It exits 1 when it finds a bad file, having already said which one,
# so that keeps going and lets the operator decide -- but only that. A blanket `|| true`
# here is what hid a stale `--dataset cryoppp` (rejected by argparse, exit 2) and a
# missing numpy for as long as it did: the annotated half was never checked and the run
# said nothing.
for id in "${ENTRIES[@]}"; do
  for scale in subset fullset; do
    status=0
    run_verify "$DL/verify_mrc_integrity.py" \
        --data-root "$DATA" --dataset "$scale" --ids "$id" || status=$?
    if [ "$status" -gt 1 ]; then
      echo "error: the integrity verifier itself failed on $id/$scale (exit $status)." >&2
      exit "$status"
    fi
  done
done

banner "Micrograph counts"
# A case rather than an associative array: macOS still ships bash 3.2.
expected_full() {  # expected_full <id>
  case "$1" in
    10081) echo 997 ;;
    10093) echo 1873 ;;
    10345) echo 1644 ;;
    10532) echo 1556 ;;
    *)     echo "?" ;;
  esac
}
count_mrc() {  # count_mrc <dir>
  if [ -d "$1" ]; then
    find "$1" -name '*.mrc' | wc -l | tr -d ' '
  else
    echo 0
  fi
}
for id in "${ENTRIES[@]}"; do
  printf "  %s  annotated %4s / 300   full %5s / %s\n" \
      "$id" \
      "$(count_mrc "$DATA/cryoppp/$id/micrographs")" \
      "$(count_mrc "$DATA/cryoppp_fullset/$id/micrographs")" \
      "$(expected_full "$id")"
done
