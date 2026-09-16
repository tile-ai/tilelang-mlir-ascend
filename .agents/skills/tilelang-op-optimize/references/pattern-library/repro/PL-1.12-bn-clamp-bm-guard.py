"""PL-1.12-bn-clamp-bm-guard -- delta repro (ED-B, delta form).

Entry: pattern-library/attention.md PL-1.12-task-pipeline-depth2
(update 2026-09-16 r9 precision_fix; the compile-fail -> compile-ok
change; effect needs the full kernel -- this file is the minimal
before/after dispatch-guard skeleton + mirrored UB accounting,
py-run-verified with assertions). Also registered for
constants.md CONST-capacity-910B2C (the mirrored UB accounting
and the 0.981 manual-vs-actual ratio recorded there).

One-line: the E6 flag-budget guard clamps bn_eff to ceil16(ceildiv(
S_kv,15)) whenever S_kv > 3840 (bn 272/288/...), and combined with the
per-shape bm=80 dispatch (half=40) the band-carry [half,288] mask chain
+ f32 ND staging overflow UB (BishengIR 210080B vs 196608B/AIV,
"--enable-auto-multi-buffer=false"); every UB buffer's leading dim is
half, so the requirement scales linearly in half and pinning the bm=80
domain to the tuned-bn domain (bn_min <= 256 -> bm=64 on the clamp
domain) restores ~28KB headroom with zero perf-domain impact (manifest
S_kv <= 2048 never leaves the tuned-bn domain; 8b-long fp16 +0.08%).

First verified: tilelang 0.1.2+a13585dc + CANN 8.5.0 + Ascend910B2C,
_gqa_prefill_fwd_kernel r9 precision_fix 2026-09-16 (full archive:
perf_opt/logs/r9_precision_fix/repro_full_fwd_bf16_r8h.log, provenance
only; manual-vs-actual ratio 0.981 recorded in the entry update).
"""

UB_BUDGET_B = 196608  # 192KB per AIV
TUNED_BN = 256
BM_LONG = 80


def bn_min_of(skv):  # E6: bn_eff >= ceil16(ceildiv(S_kv,15)) <=> nk <= 15
    return ((skv + 14) // 15 + 15) // 16 * 16


def dispatch_bm(sq, dim, skv):
    # AFTER (r9 fix): bm=80 domain additionally latched on tuned-bn domain.
    # BEFORE: the third conjunct was absent (r7e rule).
    if sq >= 256 and dim == 128 and bn_min_of(skv) <= TUNED_BN:
        return BM_LONG
    return 64


def manual_ub_b(bm, bn_eff, dim, nk, carry):
    # Mirror of the Vector-scope UB alloc list (leading dim = half each).
    h, wide, fast = bm // 2, (bn_eff if carry else 16), (16 if carry else bn_eff)
    return (
        h * bn_eff * 2
        + h * bn_eff * 4
        + h * dim * 4
        + 10 * h * 4
        + nk * h * 4
        + h * fast * 4 * 2
        + h * 4
        + h * wide * 2 * 2
        + h * wide * 4
        + h * wide * 2
    )


if __name__ == "__main__":
    # tag, Sq, dim, Skv, expected bm  (UB-checked for the D=128 wide path)
    for tag, sq, d, skv, want in [
        ("smoke-fwd", 1024, 64, 1024, 64),  # dim=64: bm=64 pre-r7e
        ("full-fwd-fp16", 2048, 128, 2048, 80),
        ("full-fwd-bf16", 4096, 128, 4096, 64),  # THE fixed case
        ("8b-short", 512, 128, 512, 80),
        ("8b-long", 2048, 128, 2048, 80),
        ("boundary-3840", 3840, 128, 3840, 80),  # bn_min=256: stays 80
        ("boundary-3841", 3841, 128, 3841, 64),
    ]:  # bn_min=272: guarded
        bm = dispatch_bm(sq, d, skv)
        assert bm == want, (tag, bm, want)
        if d == 128 and sq >= 256:
            bn = max(TUNED_BN, bn_min_of(skv), d)
            ub = manual_ub_b(bm, bn, d, (skv + bn - 1) // bn, skv % bn != 0)
            assert ub * 1.12 <= UB_BUDGET_B, (tag, ub)  # inflation allowance
            print(
                f"{tag:14s} bm={bm} bn={bn} ub={ub / 1024:.1f}KB "
                f"(x1.12 {ub * 1.12 / 1024:.1f}KB) OK"
            )
    # pre-fix failure reproduced: bm=80 on the clamped domain overflows
    ub_bad = manual_ub_b(80, 288, 128, 15, True)
    assert ub_bad > UB_BUDGET_B, ub_bad
    print(
        f"pre-fix full-fwd-bf16 (bm=80,bn=288,carry) ub={ub_bad / 1024:.1f}KB "
        f"> 192KB (BishengIR actual 205.25KB) FAIL-as-documented"
    )
    print("all assertions passed")
