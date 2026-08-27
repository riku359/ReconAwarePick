#!/usr/bin/env bash
# Release hygiene gate. Run it before every push, and certainly before the
# repository is made public.
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Release hygiene gate. Run it before every push, and certainly before the
repository is made public.

It checks the things that are cheap to check and expensive to get wrong: paths
and account names from the machines the experiments ran on, credentials, large
binaries, leftover Japanese prose, code that no longer compiles, and the
condition names from the private research repository leaking into the public
vocabulary.

  bash scripts/check_release.sh          check the working tree
  bash scripts/check_release.sh --staged only what is staged for commit
  bash scripts/check_release.sh --links  also fetch every external link

Exit status 0 means every check passed.
HELP
}

# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
# No -e: a failing check has to be reported and counted, not stop the run.
set -uo pipefail

# This script lives in scripts/, so the repository root is the directory above it.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STAGED=0
LINKS=0
for arg in "$@"; do
  case "$arg" in
    --staged)  STAGED=1 ;;
    --links)   LINKS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# Every file the checks look at: what is staged, or everything git would track
# plus anything not yet added but not ignored.
files() {
  if [ "$STAGED" -eq 1 ]; then
    git diff --cached --name-only --diff-filter=ACM
  else
    git ls-files --cached --others --exclude-standard
  fi
}

# Search every one of those files for a pattern. The file names are handed over
# NUL-separated so that a name with a space in it still arrives as one argument,
# and -H keeps the file name on every hit even when grep is given a single file.
grep_files() {  # grep_files <extended regex>
  files | tr '\n' '\0' | xargs -0 grep -HInE "$1" 2>/dev/null
}

FAIL=0

report() {  # report <label> <what was found, one per line; empty means the check passed>
  local label="$1"
  local found="${2:-}"
  if [ -z "$found" ]; then
    echo "ok    $label"
    return
  fi
  echo "FAIL  $label"
  echo "$found" | head -25 | sed 's/^/        /'
  local n
  n=$(echo "$found" | wc -l | tr -d ' ')
  if [ "$n" -gt 25 ]; then
    echo "        ... and $((n - 25)) more"
  fi
  FAIL=1
}

scan() {  # scan <label> <extended regex>
  local label="$1"
  local pattern="$2"
  local hits
  # This file names every pattern it looks for, so it is never a hit itself.
  hits=$(grep_files "$pattern" | grep -v '^scripts/check_release.sh:')
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
legacy=$(files \
  | grep -v -e '^docs/PAPER_TO_CODE.md$' -e '^results/tables/' \
  | tr '\n' '\0' \
  | xargs -0 grep -HInE 'fbf_r[0-9]|fbc_r[0-9]|general_full|lora_general|lora_chained' 2>/dev/null \
  | grep -v '^scripts/check_release.sh:')
report "no legacy condition names" "$legacy"

echo
echo "== credentials =="
scan "no CryoSPARC licence id"  'CRYOSPARC_LICENSE_ID=[^[:space:]]'
scan "no passwords in tree"     '(PASSWORD|SECRET|TOKEN|API_KEY)[[:space:]]*=[[:space:]]*[^[:space:]$"'"'"'{]'
scan "no private keys"          'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY'
report "no .env committed" "$(files | grep -E '^\.env$|/\.env$')"

echo
echo "== residue =="
# Not a grep character class: GNU grep under LC_CTYPE=POSIX compares such ranges
# byte by byte, so any multi-byte character matches, and BSD grep has no -P to fall
# back on. Python compares codepoints wherever it runs.
japanese=$(files | tr '\n' '\0' | xargs -0 python3 -c '
import sys, pathlib

# Hiragana and katakana, then the CJK ideographs. Written as numbers so that this
# file itself stays plain ASCII.
RANGES = ((0x3040, 0x30FF), (0x4E00, 0x9FFF))

def is_japanese(char):
    return any(low <= ord(char) <= high for low, high in RANGES)

for f in sys.argv[1:]:
    try:
        text = pathlib.Path(f).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for n, line in enumerate(text.splitlines(), 1):
        if any(is_japanese(char) for char in line):
            print(f"{f}:{n}:{line.strip()[:120]}")
' 2>/dev/null | grep -v '^scripts/check_release.sh:')
report "no Japanese prose"        "$japanese"
report "no .DS_Store"             "$(files | grep -F '.DS_Store')"
report "no __pycache__"           "$(files | grep -F '__pycache__')"
report "no editor swap files"     "$(files | grep -E '\.(swp|swo|orig|rej|bak)$')"

echo
echo "== size =="
big=$(files | while read -r f; do
        if [ ! -f "$f" ]; then
          continue
        fi
        kb=$(du -k "$f" 2>/dev/null | cut -f1)
        if [ -z "$kb" ]; then
          kb=0
        fi
        if [ "$kb" -gt 1024 ]; then
          printf '%6s KB  %s\n' "$kb" "$f"
        fi
      done | sort -rn)
report "no tracked file over 1 MB" "$big"
# du -c prints a "total" line last; that total, in KB, is the payload.
total=$(files | tr '\n' '\0' | xargs -0 du -ck 2>/dev/null | tail -1 | cut -f1)
if [ -z "$total" ]; then
  total=0
fi
echo "      total tracked payload: $((total / 1024)) MB"

echo
echo "== code still compiles =="
pyfail=$(files | grep '\.py$' | while read -r f; do
           python3 -m py_compile "$f" 2>/dev/null || echo "$f"
         done)
report "every .py compiles" "$pyfail"
shfail=$(files | grep '\.sh$' | while read -r f; do
           bash -n "$f" 2>/dev/null || echo "$f"
         done)
report "every .sh parses" "$shfail"

# The ambient interpreter usually has no pyyaml, so borrow one rather than
# reporting every well-formed file as broken.
if command -v uv >/dev/null; then
  YAML_READER="uv"
elif python3 -c 'import yaml' 2>/dev/null; then
  YAML_READER="python3"
else
  YAML_READER="none"
  echo "      (skipping YAML parsing: neither uv nor pyyaml is available)"
fi
yamlcheck() {  # yamlcheck <file>, non-zero when the file does not parse
  if [ "$YAML_READER" = "uv" ]; then
    uv run --quiet --with pyyaml python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$1"
  elif [ "$YAML_READER" = "python3" ]; then
    python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$1"
  fi
}
ymlfail=$(files | grep -E '\.ya?ml$' | while read -r f; do
            yamlcheck "$f" 2>/dev/null || echo "$f"
          done)
report "every .yaml parses" "$ymlfail"
jsonfail=$(files | grep '\.json$' | while read -r f; do
             python3 -c "import sys, json; json.load(open(sys.argv[1]))" "$f" 2>/dev/null || echo "$f"
           done)
report "every .json parses" "$jsonfail"

echo
echo "== documentation =="
# A dead link in a release README is the first thing a reader hits. Relative
# targets only; external URLs are not fetched.
deadlinks=$(files | grep '\.md$' | while read -r md; do
  dir="$(dirname "$md")"
  # Every ](target) in the document, with the brackets peeled off.
  grep -oE '\]\([^)#][^)]*\)' "$md" 2>/dev/null | sed 's/](//; s/)$//' | while read -r target; do
    if echo "$target" | grep -qE '^(http|mailto:)'; then
      continue
    fi
    target="$(echo "$target" | sed 's/#.*//')"       # drop the #anchor, if any
    if [ -z "$target" ]; then
      continue
    fi
    # A target that starts with / is meant from the repository root; every other
    # one is relative to the document it appears in.
    if echo "$target" | grep -q '^/'; then
      resolved="$REPO$target"
    else
      resolved="$dir/$target"
    fi
    if [ ! -e "$resolved" ]; then
      echo "$md -> $target"
    fi
  done
done)
report "no dead links between documents" "$deadlinks"

# A path quoted in prose is a promise that the file is there. These drift silently
# when something is renamed, so check them the same way as the links.
badpaths=$(files | grep -E '\.(md|ya?ml)$' | tr '\n' '\0' | xargs -0 python3 -c '
import re, sys, os, pathlib

PAT = re.compile(r"`([A-Za-z0-9_./-]*(?:src|scripts|configs|results|docs|envs)/[A-Za-z0-9_./<>-]+)`")
for f in sys.argv[1:]:
    for n, line in enumerate(pathlib.Path(f).read_text(errors="replace").splitlines(), 1):
        for m in PAT.finditer(line):
            p = m.group(1).rstrip("/")
            if "<" in p or "*" in p:            # templated, cannot be checked
                continue
            if os.path.exists(p) or os.path.exists(os.path.join(os.path.dirname(f), p)):
                continue
            print(f"{f}:{n}  {p}")
' 2>/dev/null)
report "every documented path exists" "$badpaths"

# Off by default: it needs the network, and a slow mirror should not fail a commit.
# This repository's own URL is skipped: it 404s for an anonymous fetch while the
# repository is private, which says nothing about whether the link is right.
if [ "$LINKS" -eq 1 ]; then
  echo "      checking external links (this needs the network)..."
  # Patch files are excluded: a diff records the line it removes, and a URL we
  # removed because it was dead is supposed to stay dead.
  urls=$(files \
    | grep -vE 'uv\.lock|requirements.*\.txt|environment\.lock\.yml|\.diff$' \
    | tr '\n' '\0' \
    | xargs -0 grep -ohE 'https?://[A-Za-z0-9._~:/?#@!$&()*+,;=%-]+' 2>/dev/null \
    | sed 's/[.,)`]*$//' \
    | sort -u \
    | grep -v 'github.com/riku359/ReconAwarePick')
  badurls=$(echo "$urls" | while read -r u; do
      if [ -z "$u" ]; then
        continue
      fi
      code=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 15 "$u" 2>/dev/null)
      if [ -z "$code" ]; then
        code=000
      fi
      # Anything that is not a 2xx or a 3xx is worth looking at by hand.
      if ! echo "$code" | grep -qE '^[23]'; then
        echo "$code  $u"
      fi
    done)
  report "every external link resolves" "$badurls"
fi

echo
echo "== licence and attribution =="
missing=""
if [ ! -f LICENSE ]; then
  missing="LICENSE is missing"
fi
report "LICENSE present" "$missing"
missing=""
if [ ! -f repos.lock.yaml ]; then
  missing="repos.lock.yaml is missing"
fi
report "repos.lock.yaml present" "$missing"
# Upstream code must be cloned, never committed. The one exception is the
# CryoTransformer overlay, which the LICENSE names explicitly.
report "no third_party committed" "$(files | grep '^third_party/')"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All checks passed."
else
  echo "Some checks failed. Nothing was changed; fix the entries above and re-run."
fi
exit "$FAIL"
