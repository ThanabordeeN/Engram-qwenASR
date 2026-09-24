#!/usr/bin/env bash
# Build the two Zenodo deposit archives.
#
#   ./zenodo/build_archives.sh
#
# Writes:
#   zenodo/EngramQwenASR-software-v1.0.0.zip           code, results, checkpoints  (MIT)
#   zenodo/EngramQwenASR-technical-reports-v1.0.0.zip  EN + TH reports            (CC BY 4.0)
#
# The archives are built from the working tree, not from git, because the four
# Engram checkpoints are inputs that .gitignore excludes on purpose. Everything
# else that is excluded here is excluded deliberately; see zenodo/METADATA.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/zenodo"
VERSION="${1:-1.0.0}"
cd "$ROOT"

# LaTeX intermediate files are regenerated on every build; keep them out.
rm -rf reports/build/*.aux reports/build/*.log reports/build/*.out reports/build/*.toc 2>/dev/null || true

software="$OUT/EngramQwenASR-software-v$VERSION.zip"
reports="$OUT/EngramQwenASR-technical-reports-v$VERSION.zip"
rm -f "$software" "$reports"

# --- Record 1: software, results, and the checkpoints ----------------------- #
# hf/ ships without the generated delta (hf/*.pt): that file is a strip of the
# checkpoint already archived under checkpoints/, and scripts/08 rebuilds it.
zip -q -r "$software" \
  README.md REPRODUCE.md LICENSE CITATION.cff requirements.txt .gitignore \
  src scripts configs data docs hf \
  results/predictions results/summaries results/tables results/figures \
  checkpoints \
  archive/notebooks \
  exports/nf4/manifest.json \
  -x '*/__pycache__/*' '*.pyc' 'hf/*.pt'

# --- Record 2: the technical reports, with the figures they include --------- #
# reports/*.tex resolves its figures through \graphicspath{{../../results/figures/}},
# so the archive keeps the same relative layout and the LaTeX sources compile
# as-is from a fresh extraction.
zip -q -r "$reports" \
  reports/en reports/th reports/LICENSE.md \
  results/figures \
  -x '*/__pycache__/*' '*.pyc'

echo "Built:"
for f in "$software" "$reports"; do
  printf '  %-58s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done
echo
echo "sha256:"
( cd "$OUT" && sha256sum "$(basename "$software")" "$(basename "$reports")" )
echo
echo "Verifying archive contents..."
unzip -t "$software" > /dev/null && echo "  software zip: OK"
unzip -t "$reports" > /dev/null && echo "  reports zip: OK"
