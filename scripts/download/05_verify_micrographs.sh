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
# Not fatal: there may simply be nothing to recover.
if ! run_dl "$DL/recover_failed_mrc_from_targz.py" --data-root "$DATA" --max-retries 5; then
  echo "  (nothing to recover, or recovery reported failures; see the log under cryoppp_tools/)"
fi

banner "Verifying micrograph integrity"
# The verifier exits non-zero when it finds a bad file. It has already said which one,
# so keep going and let the operator decide.
for id in "${ENTRIES[@]}"; do
  run_dl "$DL/verify_mrc_integrity.py" --data-root "$DATA" --dataset cryoppp --ids "$id" || true
  run_dl "$DL/verify_mrc_integrity.py" --data-root "$DATA" --dataset fullset --ids "$id" || true
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
