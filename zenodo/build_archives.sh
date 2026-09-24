#!/usr/bin/env bash
# Build the two Zenodo deposit archives.
#
#   ./zenodo/build_archives.sh
#
# Writes:
#   zenodo/EngramQwenASR-software-v1.0.0.zip           code + results            (MIT)
#   zenodo/EngramQwenASR-technical-reports-v1.0.0.zip  EN + TH reports            (CC BY 4.0)
#
# The Engram weights are not in either archive. They are hosted on the Hugging
# Face Hub at Thanabordee/Qwen3-ASR-0.6B-Thai-Engram and identified here by
# checkpoints/SHA256SUMS.
#
# The archives are built from the working tree, not from git, because
# checkpoints/SHA256SUMS is present while the weights it describes are not.
# Everything else that is excluded here is excluded deliberately; see
# zenodo/METADATA.md.
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

# --- Record 1: software, results, and the code ----------------------------- #
# The four Engram checkpoints are NOT in this archive. They are hosted on the
# Hugging Face Hub; checkpoints/SHA256SUMS ships instead, as the manifest of
# what those weights are. Including them would add 348 MB of duplicated weights
# and make the deposit impossible to re-upload on a slow link.
#
# hf/ ships without the generated delta (hf/*.pt): that file is a strip of the
# step-750 checkpoint and scripts/08 rebuilds it from whatever it fetches.
#
# scripts/09 is excluded: it deposits the records, so it needs zenodo/METADATA.md
# and the two zips. zenodo/ cannot ship inside the software archive, because
# METADATA.md records that archive's own sha256 and would then never match.
zip -q -r "$software" \
  README.md REPRODUCE.md LICENSE CITATION.cff requirements.txt .gitignore \
  src scripts configs data docs hf \
  results/predictions results/summaries results/tables results/figures \
  checkpoints/SHA256SUMS checkpoints/latest.json \
  archive/notebooks \
  exports/nf4/manifest.json \
  -x '*/__pycache__/*' '*.pyc' 'hf/*.pt' 'scripts/09_deposit_zenodo.py'

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
