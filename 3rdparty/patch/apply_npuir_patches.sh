#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${1:-${patch_dir}/../AscendNPU-IR}"
expected_revision=8796a8ac1508380d78128427c61ab47d03f05739
actual_revision="$(git -C "$source_dir" rev-parse HEAD)"

if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "Error: NPUIR patches require CANN 9.1.0 ($expected_revision), got $actual_revision" >&2
    exit 1
fi

for patch_file in "$patch_dir"/*.patch; do
    if git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
        echo "Already applied: $(basename "$patch_file")"
    elif git -C "$source_dir" apply --check "$patch_file"; then
        git -C "$source_dir" apply "$patch_file"
        echo "Applied: $(basename "$patch_file")"
    else
        echo "Error: Cannot apply $(basename "$patch_file"); inspect the NPUIR worktree" >&2
        exit 1
    fi
done
