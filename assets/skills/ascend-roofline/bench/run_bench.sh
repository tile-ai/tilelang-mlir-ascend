#!/bin/bash
# SIMT fp32 op-throughput probe runner. Device: 950DT, 72 AIV.
set -euo pipefail
job=$(cd "$(dirname "$0")" && pwd)
work=/home/sym/dyw/rmsnorm-tvmir-j3yJtF
source "$work/env.sh" >/dev/null 2>&1
export TILELANG_DISABLE_CACHE=0 TILELANG_CACHE_DIR="$job/cache"
export ASCEND_PROCESS_LOG_PATH="$job/ascend-logs"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$job/profiles" "$job/logs"
cd "$job"
runner="$python"

run() {
    local op=$1 rounds=$2 tag=$3 tiles=${4:-4}
    if [[ -d "$job/profiles/$tag" ]] && grep -q "^PASS op=$op " "$job/logs/$tag.log" 2>/dev/null; then
        echo "skip $tag (already captured)"
        return
    fi
    rm -rf "$job/profiles/$tag"
    if msprof op --aic-metrics=Default,Roofline,Occupancy --kernel-name=main_kernel \
        --output="$job/profiles/$tag" "$runner" -u "$job/bench_vector_ops.py" "$op" \
        --rounds "$rounds" --tiles "$tiles" >"$job/logs/$tag.log" 2>&1 \
        && grep -q "^PASS op=$op " "$job/logs/$tag.log"; then
        echo "ok   $tag $(grep -oE 'Task Duration\(us\): [0-9.]+' "$job/logs/$tag.log" || true)"
    else
        echo "FAIL $tag (see logs/$tag.log)"
        tail -5 "$job/logs/$tag.log" || true
    fi
}

case ${1:-full} in
verify)
    for op in exp ln sqrt rsqrt rcp sigmoid; do
        if "$runner" -u "$job/bench_vector_ops.py" "$op" --rounds 1 --verify >"$job/logs/verify-$op.log" 2>&1; then
            echo "verify ok   $op"
        else
            echo "verify FAIL $op"; tail -5 "$job/logs/verify-$op.log"
        fi
    done
    ;;
extra)
    if "$runner" -u "$job/bench_vector_ops.py" expdif --rounds 1 --verify >"$job/logs/verify-expdif.log" 2>&1; then
        echo "verify ok   expdif"
    else
        echo "verify FAIL expdif"; tail -5 "$job/logs/verify-expdif.log"
    fi
    run expdif 128 expdif-r128
    run expdif 128 expdif-r128-b
    ;;
range)
    # Same-semantics writes differ by up to 2.5x; capture the bracket explicitly.
    for op in expm exps; do
        if "$runner" -u "$job/bench_vector_ops.py" "$op" --rounds 1 --verify >"$job/logs/verify-$op.log" 2>&1; then
            echo "verify ok   $op"
        else
            echo "verify FAIL $op"; tail -5 "$job/logs/verify-$op.log"
        fi
        run "$op" 128 "$op-r128"
    done
    ;;
smoke)
    run control 128 control-r128
    run exp 128 exp-r128
    ;;
full)
    run fma 512 fma-r512
    run fma 2048 fma-r2048
    for r in 64 128 256; do
        run control "$r" "control-r$r"
        run exp "$r" "exp-r$r"
    done
    for op in ln sqrt rsqrt rcp sigmoid; do
        run "$op" 128 "$op-r128"
    done
    run control 128 control-r128-t8 8
    run exp 128 exp-r128-t8 8
    run control 128 control-r128-b
    run exp 128 exp-r128-b
    ;;
esac
