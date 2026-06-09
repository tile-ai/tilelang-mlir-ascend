#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

EXAMPLES_DIR = Path(__file__).resolve().parent

SKIP_FILES = {
    "run_all.py",
    "torch_tl_ops/compile/kernels/gemm.py",
    "torch_tl_ops/compile/kernels/flash_attention.py",
    "torch_tl_ops/compile/kernels/__init__.py",
    "torch_tl_ops/compile/precompile.py",
    "torch_tl_ops/setup.py",
    "torch_tl_ops/src/ops/base.py",
    "torch_tl_ops/src/ops/gemm.py",
    "torch_tl_ops/src/ops/flash_attention.py",
    "torch_tl_ops/src/ops/__init__.py",
    "torch_tl_ops/src/kernels/__init__.py",
    "torch_tl_ops/src/utils/__init__.py",
    "torch_tl_ops/src/loader.py",
    "torch_tl_ops/src/registry.py",
    "torch_tl_ops/src/__init__.py",
    "torch_tl_ops/tests/__init__.py",
    "torch_tl_ops/tests/test_flash_attention.py",
    "torch_tl_ops/tests/test_gemm.py",
}

SKIP_DIRS = {
    "torch_tl_ops/src",
    "torch_tl_ops/compile",
    "deepseek_v4/inference",
}


@dataclass
class RunResult:
    name: str
    status: str
    duration: float = 0.0
    error_msg: str = ""
    device_id: int = -1


def get_npu_device_count():
    try:
        import torch

        count = torch.npu.device_count()
        if count > 0:
            return count
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import re

            matches = re.findall(r"NPU\s+(\d+)", result.stdout)
            if matches:
                return len(set(matches))
    except Exception:
        pass
    return 1


def collect_examples(base_dir: Path):
    examples = []
    for py_file in sorted(base_dir.rglob("*.py")):
        rel = py_file.relative_to(base_dir)
        rel_str = str(rel)

        if rel.name == "__init__.py":
            continue
        if rel_str in SKIP_FILES:
            continue
        if any(rel_str.startswith(d) for d in SKIP_DIRS):
            continue

        content = py_file.read_text(errors="ignore")
        if "if __name__" not in content and "def test_" not in content:
            continue

        examples.append(rel_str)
    return examples


def run_example(rel_path: str, base_dir: Path, device_id: int, timeout: int = 300):
    abs_path = base_dir / rel_path
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(abs_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(base_dir),
            env=env,
        )
        duration = time.time() - start
        if proc.returncode == 0:
            return RunResult(rel_path, "PASSED", duration, "", device_id)
        else:
            stderr_lines = proc.stderr.strip().split("\n")
            tail = (
                "\n".join(stderr_lines[-5:])
                if len(stderr_lines) > 5
                else proc.stderr.strip()
            )
            return RunResult(rel_path, "FAILED", duration, tail, device_id)
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return RunResult(
            rel_path, "TIMEOUT", duration, f"Exceeded {timeout}s", device_id
        )
    except Exception as e:
        duration = time.time() - start
        return RunResult(rel_path, "ERROR", duration, str(e), device_id)


def print_header(examples, num_devices):
    width = max(len(e) for e in examples) + 2
    width = max(width, 40)
    print(f"{'Example':<{width}} {'Result':<10} {'Time':>8} {'NPU':>4}")
    print("-" * (width + 26))
    return width


def print_result(result: RunResult, width: int):
    status_colors = {
        "PASSED": "\033[92m",
        "FAILED": "\033[91m",
        "TIMEOUT": "\033[93m",
        "ERROR": "\033[91m",
    }
    reset = "\033[0m"
    color = status_colors.get(result.status, "")
    duration_str = f"{result.duration:.2f}s"
    device_str = f"NPU{result.device_id}"
    print(
        f"{result.name:<{width}} {color}{result.status:<10}{reset} {duration_str:>8} {device_str:>4}"
    )
    if result.error_msg and result.status != "PASSED":
        for line in result.error_msg.split("\n"):
            print(f"{'':>{width}}   {color}{line}{reset}")


def print_summary(results: list[RunResult]):
    passed = sum(1 for r in results if r.status == "PASSED")
    failed = sum(1 for r in results if r.status == "FAILED")
    timeout = sum(1 for r in results if r.status == "TIMEOUT")
    errored = sum(1 for r in results if r.status == "ERROR")
    total = len(results)
    total_time = sum(r.duration for r in results)

    print("\n" + "=" * 60)
    print(
        f"Results: {passed} passed, {failed} failed, {timeout} timeout, {errored} error"
    )
    print(f"Total: {total} examples in {total_time:.2f}s")

    if failed + timeout + errored > 0:
        print("\nFailed examples:")
        for r in results:
            if r.status != "PASSED":
                print(f"  {r.status}: {r.name} (NPU{r.device_id})")

    print("=" * 60)
    return failed + timeout + errored == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all TileLang NPU examples")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout per example (seconds)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run examples matching this substring",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: NPU device count)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially on NPU0 (for debugging)",
    )
    args = parser.parse_args()

    num_devices = get_npu_device_count()

    if args.sequential:
        max_workers = 1
    else:
        max_workers = args.workers if args.workers is not None else num_devices

    examples = collect_examples(EXAMPLES_DIR)
    if not examples:
        print("No examples found.")
        sys.exit(1)

    if args.filter:
        examples = [e for e in examples if args.filter in e]
        if not examples:
            print(f"No examples matching filter: {args.filter}")
            sys.exit(1)

    assignments = [(e, i % num_devices) for i, e in enumerate(examples)]

    print(
        f"Collected {len(examples)} examples, {num_devices} NPU device(s), {max_workers} worker(s)\n"
    )
    width = print_header(examples, num_devices)

    results = [None] * len(examples)
    print_lock = threading.Lock()
    next_print_idx = [0]
    fail_fast_event = threading.Event()

    def run_task(idx, rel_path, device_id):
        if fail_fast_event.is_set():
            results[idx] = RunResult(rel_path, "SKIPPED", 0.0, "", device_id)
            with print_lock:
                _flush_ordered(results, width)
            return

        result = run_example(rel_path, EXAMPLES_DIR, device_id, timeout=args.timeout)
        results[idx] = result

        with print_lock:
            _flush_ordered(results, width)

        if args.fail_fast and result.status != "PASSED":
            fail_fast_event.set()

    def _flush_ordered(results_list, w):
        while (
            next_print_idx[0] < len(results_list)
            and results_list[next_print_idx[0]] is not None
        ):
            print_result(results_list[next_print_idx[0]], w)
            next_print_idx[0] += 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (rel_path, device_id) in enumerate(assignments):
            f = executor.submit(run_task, idx, rel_path, device_id)
            futures.append(f)
        for f in futures:
            f.result()

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
