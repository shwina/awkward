#!/usr/bin/env bash
# Run the public-API benchmark on a "before" commit (pre cuda.compute migration)
# and an "after" commit (current main, or awkward3 branch with more kernels migrated),
# then produce a comparison report.
#
# Usage:
#   ./run_compare.sh BEFORE_REF AFTER_REF [-- extra-bench-args...]
#
# Examples:
#   # Compare pre-everything (no cuda.compute) vs current main:
#   ./run_compare.sh c7e21ea0 main
#
#   # Compare current main vs awkward3 (latest migration branch):
#   ./run_compare.sh main upstream/awkward3
#
# Notes:
#   * This script does git checkouts on the working tree. Make sure the
#     working tree is clean and the script itself is staged/committed
#     somewhere safe, OR run it from a separate worktree, e.g.:
#       git worktree add /tmp/awkward-bench main
#   * The "before" commit must have ak.{sum,max,argmin,...} working on GPU
#     via the legacy .cu kernel path. ak.sort on GPU only exists post-migration.
#   * Re-builds awkward-cpp if needed; this can take several minutes.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 BEFORE_REF AFTER_REF [-- extra-bench-args...]" >&2
    exit 1
fi

BEFORE_REF="$1"
AFTER_REF="$2"
shift 2
if [ "${1:-}" = "--" ]; then shift; fi
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="${REPO_ROOT}/studies/cuda-compute-migration"
OUT_DIR="${BENCH_DIR}/results"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

# Save the original HEAD so we can restore it.
ORIG_HEAD=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)

# Confirm clean working tree.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree is dirty. Commit/stash changes before running." >&2
    exit 1
fi

# Re-install awkward in-place after each checkout.
reinstall() {
    echo "[run_compare] reinstalling awkward (this may take a few minutes)..."
    pip install -e . --no-build-isolation -q
}

run_bench_at() {
    local ref="$1"
    local out_file="$2"
    echo "[run_compare] checking out $ref"
    git checkout "$ref"
    reinstall
    echo "[run_compare] running bench at $ref"
    python "${BENCH_DIR}/bench_public_apis.py" \
        --output "$out_file" \
        --label "$ref" \
        "${EXTRA_ARGS[@]}"
}

cleanup() {
    echo "[run_compare] restoring $ORIG_HEAD"
    git checkout "$ORIG_HEAD" >/dev/null 2>&1 || true
}
trap cleanup EXIT

BEFORE_JSON="${OUT_DIR}/bench_before.json"
AFTER_JSON="${OUT_DIR}/bench_after.json"

run_bench_at "$BEFORE_REF" "$BEFORE_JSON"
run_bench_at "$AFTER_REF"  "$AFTER_JSON"

# After running, restore original branch (via trap), then produce a comparison.
git checkout "$ORIG_HEAD"
echo
echo "==================== COMPARISON ===================="
python "${BENCH_DIR}/compare.py" "$BEFORE_JSON" "$AFTER_JSON" \
       --markdown > "${OUT_DIR}/report.md"
cat "${OUT_DIR}/report.md"
echo
echo "Results: $BEFORE_JSON, $AFTER_JSON"
echo "Report : ${OUT_DIR}/report.md"
