#!/usr/bin/env bash
# Release hygiene gate. Run it before every push, and certainly before the
# repository is made public.
#
# It checks the things that are cheap to check and expensive to get wrong: paths
# and account names from the machines the experiments ran on, credentials, large
# binaries, leftover Japanese prose, code that no longer compiles, and the
# condition names from the private research repository leaking into the public
# vocabulary.
#
#   bash scripts/check_release.sh          check the working tree
#   bash scripts/check_release.sh --staged only what is staged for commit
#
# Exit status 0 means every check passed.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGED=0
[ "${1:-}" = "--staged" ] && STAGED=1

if [ "$STAGED" -eq 1 ]; then
  files() { git diff --cached --name-only --diff-filter=ACM; }
else
  # Everything git would track, plus anything not yet added but not ignored.
  files() { git ls-files --cached --others --exclude-standard; }
fi

FAIL=0
report() {  # report <label> <matches...>
  local label="$1"; shift
  if [ -n "${1:-}" ]; then
    echo "FAIL  $label"
    printf '%s\n' "$@" | sed 's/^/        /' | head -25
    local n; n=$(printf '%s\n' "$@" | wc -l | tr -d ' ')
    [ "$n" -gt 25 ] && echo "        ... and $((n - 25)) more"
    FAIL=1
  else
    echo "ok    $label"
  fi
}

scan() {  # scan <label> <regex>
  local label="$1" pattern="$2" hits
  hits=$(files | tr '\n' '\0' | xargs -0 grep -InE "$pattern" 2>/dev/null | grep -v '^scripts/check_release.sh:')
  report "$label" "$hits"
}

echo "== paths and hosts from the experiment machines =="
scan "no /home/riku"            '/home/riku'
scan "no HPC account name"      'v_riku_itsuji'
scan "no local macOS paths"     '/Users/[A-Za-z]'
# `rapick-data` on its own is fine: docs use it as an example directory name.
# What must not survive is a real root from the experiment machines. `orient_diag`
# is deliberately absent too: it is an internal job-step name, not a path.
scan "no shared_data roots"     'shared_data|/shared/home|rapick_experiments|rapick_results'
scan "no lab disks"             '/mnt/(hdd|ssd)'
scan "no host names"            'dlbox[0-9]*|DL-BoxVI|xu-lab|xulab|aces-jump|hprc\.tamu'
scan "no sibling repo paths"    'Desktop/[a-z0-9]*2027|Desktop/recon-aware-pick|recon-pipe/'
scan "no venue name"            'WACV|wacv'

echo
echo "== the private repository's vocabulary must not leak =="
# The research repo names conditions differently. Two places may still carry the
# old names: docs/PAPER_TO_CODE.md, which is the translation table, and
# results/tables/, where they appear inside provenance fields recording which
# run each number came from. Everywhere else they are a leak.
#
# Two strings are deliberately NOT in this pattern. `cryotransformer_clean_tri` is
# the file the contamination filter writes, not a condition name, and it is the
# name the published Hugging Face artifacts carry; src/rapick/cleaner/README.md
# explains the difference between it and the condition name `mask`. And `fbgt_r`
# is the loop's source prefix for the `fb_gt` arm, formed the same way as `fb_r`
# and `fbnm_r` from the release's own condition names.
hits=$(files | grep -v -e '^docs/PAPER_TO_CODE.md$' -e '^results/tables/' | tr '\n' '\0' \
       | xargs -0 grep -InE 'fbf_r[0-9]|fbc_r[0-9]|general_full|lora_general|lora_chained' 2>/dev/null \
       | grep -v '^scripts/check_release.sh:')
report "no legacy condition names" "$hits"

echo
echo "== credentials =="
scan "no CryoSPARC licence id"  'CRYOSPARC_LICENSE_ID=[^[:space:]]'
scan "no passwords in tree"     '(PASSWORD|SECRET|TOKEN|API_KEY)[[:space:]]*=[[:space:]]*[^[:space:]$"'"'"'{]'
scan "no private keys"          'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY'
report "no .env committed" "$(files | grep -E '^\.env$|/\.env$' || true)"

echo
echo "== residue =="
scan "no Japanese prose"        '[ぁ-んァ-ヶ一-龯]'
report "no .DS_Store"           "$(files | grep -F '.DS_Store' || true)"
report "no __pycache__"         "$(files | grep -F '__pycache__' || true)"
report "no editor swap files"   "$(files | grep -E '\.(swp|swo|orig|rej|bak)$' || true)"

echo
echo "== size =="
big=$(files | while read -r f; do
        [ -f "$f" ] || continue
        kb=$(du -k "$f" 2>/dev/null | cut -f1)
        [ "${kb:-0}" -gt 1024 ] && printf '%6s KB  %s\n' "$kb" "$f"
      done | sort -rn)
report "no tracked file over 1 MB" "$big"
total=$(files | tr '\n' '\0' | xargs -0 du -ck 2>/dev/null | tail -1 | cut -f1)
echo "      total tracked payload: $(( ${total:-0} / 1024 )) MB"

echo
echo "== code still compiles =="
pyfail=$(files | grep '\.py$' | while read -r f; do
           python3 -m py_compile "$f" 2>/dev/null || echo "$f"
         done)
report "every .py compiles" "$pyfail"
shfail=$(files | grep '\.sh$' | while read -r f; do bash -n "$f" 2>/dev/null || echo "$f"; done)
report "every .sh parses" "$shfail"
# The ambient interpreter usually has no pyyaml, so borrow one rather than
# reporting every well-formed file as broken.
if command -v uv >/dev/null; then
  yamlcheck() { uv run --quiet --with pyyaml python3 -c "import sys,yaml;yaml.safe_load(open(sys.argv[1]))" "$1"; }
elif python3 -c 'import yaml' 2>/dev/null; then
  yamlcheck() { python3 -c "import sys,yaml;yaml.safe_load(open(sys.argv[1]))" "$1"; }
else
  yamlcheck() { return 0; }
  echo "      (skipping YAML parsing: neither uv nor pyyaml is available)"
fi
ymlfail=$(files | grep -E '\.ya?ml$' | while read -r f; do
            yamlcheck "$f" 2>/dev/null || echo "$f"
          done)
report "every .yaml parses" "$ymlfail"
jsonfail=$(files | grep '\.json$' | while read -r f; do
             python3 -c "import sys,json;json.load(open(sys.argv[1]))" "$f" 2>/dev/null || echo "$f"
           done)
report "every .json parses" "$jsonfail"

echo
echo "== documentation =="
# A dead link in a release README is the first thing a reader hits. Relative
# targets only; external URLs are not fetched.
deadlinks=$(files | grep '\.md$' | while read -r md; do
  dir="$(dirname "$md")"
  grep -oE '\]\([^)#][^)]*\)' "$md" 2>/dev/null | sed 's/](//;s/)$//' | while read -r target; do
    # The leading ( is required: inside $( ), a bare ) in a case pattern closes
    # the command substitution.
    case "$target" in (http*|mailto:*) continue ;; esac
    target="${target%%#*}"
    [ -z "$target" ] && continue
    if [ "${target#/}" != "$target" ]; then resolved="$REPO$target"; else resolved="$dir/$target"; fi
    [ -e "$resolved" ] || echo "$md -> $target"
  done
done)
report "no dead links between documents" "$deadlinks"

echo
echo "== licence and attribution =="
report "LICENSE present"        "$([ -f LICENSE ] || echo 'LICENSE is missing')"
report "repos.lock.yaml present" "$([ -f repos.lock.yaml ] || echo 'repos.lock.yaml is missing')"
# Upstream code must be cloned, never committed. The one exception is the
# CryoTransformer overlay, which the LICENSE names explicitly.
report "no third_party committed" "$(files | grep '^third_party/' || true)"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All checks passed."
else
  echo "Some checks failed. Nothing was changed; fix the entries above and re-run."
fi
exit "$FAIL"
