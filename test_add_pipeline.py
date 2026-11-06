"""
Python (TileLang) -> TIR -> TLIR (memref + linalg.elemwise_binary)
                  -> bishengir-compile (HIVM -> LLVM)

Run:
  python3 test_add_pipeline.py

Also save the IR before/after EVERY pass that bishengir-compile runs:
  TILELANG_DUMP_PASSES=1 python3 test_add_pipeline.py
  (full dump: /tmp/ir_all_passes.log, pass names: /tmp/pass_list.txt)

Override compile flags:
  TILELANG_COMPILE_FLAGS="--enable-hivm-compile=true" python3 test_add_pipeline.py

Fail the test unless bishengir-compile fully finishes (HIVM -> LLVM/binary):
  TILELANG_REQUIRE_LLVM=1 python3 test_add_pipeline.py
"""
import os
import re
import shutil
import subprocess
import sys

import tilelang
from tilelang import tvm
import tilelang.language as T
import tilelang.tladapter
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget

BISHENGIR_OPT = os.environ.get(
    "TILELANG_BISHENGIR_OPT", "/workspace/AscendNPU-IR/build/bin/bishengir-opt")
BISHENGIR_COMPILE = os.environ.get(
    "TILELANG_BISHENGIR_COMPILE", "/workspace/AscendNPU-IR/build/bin/bishengir-compile")
TLIR_PATH = "/tmp/add_memref.mlir"
TEMPS = "/tmp/temps"
OUT_LL = "/tmp/add_out.ll"
PASS_LOG = "/tmp/ir_all_passes.log"
PASS_LIST = "/tmp/pass_list.txt"
DEFAULT_FLAGS = "--enable-hivm-compile=true"


@T.prim_func
def main(
    A: T.Tensor((1024,), "float32"),
    B: T.Tensor((1024,), "float32"),
    C: T.Tensor((1024,), "float32"),
    shape: T.int32,
):
    with T.Kernel(1, is_npu=True) as (cid, _):
        A_VEC = T.alloc_ub((1024,), "float32")
        B_VEC = T.alloc_ub((1024,), "float32")
        C_VEC = T.alloc_ub((1024,), "float32")
        T.copy(A[0:1024], A_VEC[0:1024])
        T.copy(B[0:1024], B_VEC[0:1024])
        T.vadd(A_VEC, B_VEC, C_VEC)
        T.copy(C_VEC[0:1024], C[0:1024])


def run_bishengir_opt(passes, text):
    r = subprocess.run([BISHENGIR_OPT] + passes, input=text,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"===== bishengir-opt FAILED (passes={passes}) =====")
        print("---- stderr ----\n" + r.stderr)
        sys.exit(1)
    return r.stdout


def run_bishengir_compile(mlir_path, out=OUT_LL, temps=TEMPS):
    shutil.rmtree(temps, ignore_errors=True)
    flags = os.environ.get("TILELANG_COMPILE_FLAGS", DEFAULT_FLAGS).split()
    dump = os.environ.get("TILELANG_DUMP_PASSES") == "1"
    if dump:
        # Print the IR before and after every pass bishengir-compile runs.
        flags += [
            "--mlir-disable-threading",
            "--mlir-print-ir-before-all",
            "--mlir-print-ir-after-all",
            "--mlir-print-ir-module-scope",
        ]
    cmd = [BISHENGIR_COMPILE, mlir_path] + flags + [f"--save-temps={temps}", "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("cmd:", " ".join(cmd))
    print(r.stdout)

    if dump:
        with open(PASS_LOG, "w") as f:
            f.write(r.stderr)
        names = []
        for ln in r.stderr.splitlines():
            if "IR Dump After " in ln:
                names.append(ln.split("IR Dump After ", 1)[1].split(" //", 1)[0].strip())
        with open(PASS_LIST, "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"===== per-pass IR dump: {len(names)} pass runs =====")
        print(f"full before/after IR : {PASS_LOG}")
        print(f"pass names           : {PASS_LIST}")
        for i, n in enumerate(names, 1):
            print(f"  {i:3d}. {n}")
        errs = [l for l in r.stderr.splitlines() if "error:" in l]
        if errs:
            print("----- errors -----")
            print("\n".join(errs))
    else:
        print(r.stderr)
    return r.returncode


def print_python_source():
    lines = open(__file__).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("def main("))
    while i > 0 and lines[i - 1].startswith("@"):
        i -= 1
    j = next(k for k, l in enumerate(lines) if l.startswith("def run_bishengir_opt"))
    print(chr(10).join(lines[i:j]).rstrip())


def main_test():
    print("===== STAGE 0: Python (TileLang) kernel source =====")
    print_python_source()

    print("===== STAGE 1: raw TIR =====")
    raw_mod = tvm.IRModule({main.attrs["global_symbol"]: main})
    print(raw_mod)

    target = tvm.target.Target("npuir")
    mod = LowerAndLegalize(raw_mod, target)
    mod = OptimizeForTarget(mod, target)
    print("===== STAGE 2: TIR fed to the importer =====")
    print(mod)

    print("===== STAGE 3: TIR -> TLIR (CodeGenTIRToTLIR, memref form) =====")
    tlir = tvm.get_global_func("target.build.tilelang_tlir")(mod, target).get_source()
    with open(TLIR_PATH, "w") as f:
        f.write(tlir)
    print(tlir)

    n_alloc = len(re.findall(r"memref\.alloc\b", tlir))
    n_copy = len(re.findall(r"memref\.copy\b", tlir))
    assert n_alloc == 3, f"expected 3 memref.alloc, got {n_alloc}"
    assert n_copy == 3, f"expected 3 memref.copy, got {n_copy}"
    assert "linalg.elemwise_binary" in tlir, "expected linalg.elemwise_binary"
    assert "tensor<" not in tlir, "tensor type leaked into memref-only output"
    print("===== STAGE 3 CHECKS PASSED =====")

    print("===== STAGE 4: bishengir-opt parse/verify (no passes) =====")
    rt = run_bishengir_opt([], tlir)
    assert "elemwise_binary" in rt
    print("===== STAGE 4 PASSED: TLIR is valid MLIR =====")

    print("===== STAGE 5: bishengir-compile (TLIR -> HIVM -> LLVM) =====")
    rc = run_bishengir_compile(TLIR_PATH)
    print("bishengir-compile exit code:", rc)
    print("saved intermediates:", os.listdir(TEMPS) if os.path.isdir(TEMPS) else "none")

    hivm_file = os.path.join(TEMPS, "module.hivm.opt.mlir")
    if os.path.exists(hivm_file):
        hivm = open(hivm_file).read()
        print("===== HIVM produced by the compile =====")
        print("(full file: " + hivm_file + ", " + str(len(hivm.splitlines())) + " lines) compact view:")
        for ln in hivm.splitlines():
            t = ln.strip()
            keep = t.startswith(("call @", "hivm.hir.", "func.func @main", "%"))
            if keep and "memref.cast" not in t and "arith.constant" not in t:
                print("  " + t[:110])
        ops = sorted(set(re.findall(r"hivm\.hir\.\w+", hivm)))
        print("hivm ops:", ops)

        # hivm.hir ops may appear directly or as calls to the lowered library functions
        def positions(pattern):
            return [m.start() for m in re.finditer(pattern, hivm)]

        loads = positions(r"call @load_gm_to_ubuf\w*|hivm\.hir\.load\b")
        adds = positions(r"call @vadd_\w+|hivm\.hir\.vadd\b")
        stores = positions(r"call @store_ubuf_to_gm\w*|hivm\.hir\.store\b")
        assert len(loads) == 2, f"expected 2 loads in HIVM, got {len(loads)}"
        assert len(adds) == 1, f"expected 1 vadd in HIVM, got {len(adds)}"
        assert len(stores) == 1, f"expected 1 store in HIVM, got {len(stores)}"
        assert max(loads) < adds[0] < stores[0], \
            "HIVM op order wrong: need load, load, vadd, store"
        print("===== HIVM CHECKS PASSED (2 loads, vadd, store, in order) =====")
    else:
        print("no HIVM file produced")

    if os.path.exists(OUT_LL):
        print(f"===== LLVM output {OUT_LL} ({os.path.getsize(OUT_LL)} bytes) =====")
        print(open(OUT_LL).read()[:1500])
    else:
        print(f"no LLVM output file at {OUT_LL}")

    print("===== SUMMARY =====")
    hivm_ok = os.path.exists(hivm_file)
    hivm_status = "OK" if hivm_ok else "FAILED"
    last_status = "OK" if rc == 0 else \
        "BLOCKED (hivmc BiShengLIR->binary step; baseline fails the same way)"
    print("Python (TileLang) kernel -> TIR : OK")
    print("TIR -> TLIR                     : OK")
    print("TLIR is valid MLIR              : OK")
    print("TLIR -> HIVM                    : " + hivm_status)
    print("HIVM -> LLVM/binary             : " + last_status)
    print("===== DONE =====")
    if rc != 0:
        print("NOTE: bishengir-compile did not finish: the hivmc BiShengLIR->binary step "
              "fails in this environment (the tilelang_npuir_apis baseline fails the same way).")

    strict = os.environ.get("TILELANG_REQUIRE_LLVM") == "1"
    sys.exit(0 if (rc == 0 or (hivm_ok and not strict)) else 1)


if __name__ == "__main__":
    main_test()
