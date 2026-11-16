#!/bin/bash
# SIMD probe: verify then (optionally) profile vexpdif / vexp / control.
set -euo pipefail
job=/home/sym/dyw/roofline-bench
work=/home/sym/dyw/rmsnorm-tvmir-j3yJtF
source "$work/env.sh" >/dev/null 2>&1
export TILELANG_DISABLE_CACHE=0 TILELANG_CACHE_DIR="$job/cache"
cd "$job"
mkdir -p logs profiles

verify() {
    "$python" -u "$job/bench_vector_simd.py" "$1" --rounds 1 --verify >"logs/verify-$1.log" 2>&1
}

run() {
    local op=$1 rounds=$2 tag=$3
    rm -rf "profiles/$tag"
    if msprof op --aic-metrics=Default,Roofline,Occupancy --kernel-name=main_kernel \
        --output="$job/profiles/$tag" "$python" -u "$job/bench_vector_simd.py" "$op" --rounds "$rounds" \
        >"logs/$tag.log" 2>&1 && grep -q "^PASS op=$op " "logs/$tag.log"; then
        echo "ok   $tag $(grep -oE 'Task Duration\(us\): [0-9.]+' "logs/$tag.log" || true)"
    else
        echo "FAIL $tag"
        tail -6 "logs/$tag.log"
    fi
}

case ${1:-verify} in
verify)
    for op in vexpdif vexp control; do
        if verify "$op"; then echo "verify ok   $op"; else echo "verify FAIL $op"; tail -4 "logs/verify-$op.log"; fi
    done
    ;;
profile)
    run control 32 simdcontrol-r32
    run vexpdif 32 vexpdif-r32
    run vexp 32 vexp-r32
    ;;
esac
