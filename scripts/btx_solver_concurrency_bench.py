#!/usr/bin/env python3
"""
Local BTX solver concurrency benchmark harness.

This script compares one solver process versus multiple concurrent solver
processes per GPU. It is intentionally local-only: it does not SSH, call RPC,
start or stop miners, edit configs, or touch live rigs. It only starts local
copies of btx-matmul-solve-bench or btx-gbt-solve with per-process environment
variables and writes benchmark artifacts under the output directory.

Typical use from the repo root:

  python3 scripts/btx_solver_concurrency_bench.py \
    --build-dir build-cuda \
    --solver matmul-bench \
    --gpus 0 \
    --processes-per-gpu 1,2 \
    --tries 4194304 \
    --iterations 3 \
    --lookahead-values 0,1 \
    --batch-sizes 4 \
    --pool-slots 6 \
    --solver-threads 5

For the synthetic getblocktemplate-style solver path:

  python3 scripts/btx_solver_concurrency_bench.py \
    --build-dir build-cuda \
    --solver gbt-solve \
    --gpus 0 \
    --processes-per-gpu 1,2 \
    --tries 4194304 \
    --batch-sizes 4 \
    --pool-slots 6 \
    --solver-threads 5

Useful flags:

  --dry-run                 Print the command matrix without launching solvers.
  --binary PATH             Use an explicit solver binary instead of auto-find.
  --gpus 0,1                Local CUDA ordinals to test. Use "auto" for nvidia-smi.
  --lookahead-values 0,1    Values for BTX_MATMUL_NONCE_SEED_LOOKAHEAD.
  --batch-sizes auto,4      auto leaves BTX_MATMUL_SOLVE_BATCH_SIZE unset.
  --pool-slots auto,6       auto leaves BTX_MATMUL_CUDA_POOL_SLOTS unset.
  --solver-threads auto,5   auto leaves BTX_MATMUL_SOLVER_THREADS unset.

Outputs:

  <out-dir>/<timestamp>/summary.json
  <out-dir>/<timestamp>/summary.csv
  <out-dir>/<timestamp>/<case>/proc-*.stdout
  <out-dir>/<timestamp>/<case>/proc-*.stderr
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MATMUL_BINARY = "btx-matmul-solve-bench"
GBT_BINARY = "btx-gbt-solve"
HEX64_ZERO = "0" * 64


@dataclass(frozen=True)
class SweepCase:
    processes_per_gpu: int
    lookahead: str
    batch_size: str
    pool_slots: str
    solver_threads: str
    repeat: int


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    gpu: str
    local_index: int
    nonce_start: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv_values(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    return values


def parse_positive_csv(text: str) -> list[int]:
    values: list[int] = []
    for raw in parse_csv_values(text):
        try:
            value = int(raw, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer: {raw}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"value must be positive: {raw}")
        values.append(value)
    return values


def is_auto(value: str) -> bool:
    return value.lower() in {"auto", "unset", "default", "-"}


def exe_names(name: str) -> list[str]:
    if platform.system().lower() == "windows":
        return [f"{name}.exe", name]
    return [name, f"{name}.exe"]


def find_binary(args: argparse.Namespace) -> Path:
    if args.binary:
        path = Path(args.binary).expanduser()
        if not path.exists():
            raise SystemExit(f"error: --binary does not exist: {path}")
        return path

    binary_base = MATMUL_BINARY if args.solver == "matmul-bench" else GBT_BINARY
    root = repo_root()
    candidates: list[Path] = []

    if args.build_dir:
        build_dir = Path(args.build_dir)
        if not build_dir.is_absolute():
            build_dir = root / build_dir
        for name in exe_names(binary_base):
            candidates.extend([
                build_dir / "bin" / name,
                build_dir / "src" / name,
                build_dir / name,
            ])
    else:
        for build_dir in sorted(root.glob("build*")):
            if not build_dir.is_dir():
                continue
            for name in exe_names(binary_base):
                candidates.append(build_dir / "bin" / name)
                candidates.append(build_dir / "src" / name)
                candidates.append(build_dir / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    path_hit = shutil.which(binary_base)
    if path_hit:
        return Path(path_hit)

    hint = "--build-dir <dir> or --binary <path>"
    raise SystemExit(f"error: could not find {binary_base}; pass {hint}")


def detect_gpus(gpu_arg: str) -> list[str]:
    if gpu_arg.lower() != "auto":
        return parse_csv_values(gpu_arg)

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise SystemExit("error: --gpus auto requires nvidia-smi on PATH")
    proc = subprocess.run(
        [nvidia_smi, "-L"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(f"error: nvidia-smi -L failed: {proc.stderr.strip()}")
    gpus = []
    for line in proc.stdout.splitlines():
        match = re.match(r"GPU\s+(\d+):", line)
        if match:
            gpus.append(match.group(1))
    if not gpus:
        raise SystemExit("error: nvidia-smi did not report any GPUs")
    return gpus


def add_optional_env(env: dict[str, str], key: str, value: str) -> None:
    if not is_auto(value):
        env[key] = value


def worker_env(args: argparse.Namespace, case: SweepCase, worker: WorkerSpec) -> dict[str, str]:
    env = os.environ.copy()
    env["BTX_MATMUL_BACKEND"] = args.backend
    if args.require_backend:
        env["BTX_MATMUL_REQUIRE_BACKEND"] = args.backend
    env["BTX_MATMUL_GPU_INPUTS"] = args.gpu_inputs
    env["BTX_MATMUL_CUDA_DEVICES"] = worker.gpu
    env["BTX_MATMUL_NONCE_SEED_LOOKAHEAD"] = case.lookahead
    add_optional_env(env, "BTX_MATMUL_SOLVE_BATCH_SIZE", case.batch_size)
    add_optional_env(env, "BTX_MATMUL_CUDA_POOL_SLOTS", case.pool_slots)
    add_optional_env(env, "BTX_MATMUL_SOLVER_THREADS", case.solver_threads)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = worker.gpu
    return env


def maybe_add(cmd: list[str], flag: str, value: str) -> None:
    if not is_auto(value):
        cmd.extend([flag, value])


def build_matmul_command(binary: Path, args: argparse.Namespace, case: SweepCase) -> list[str]:
    cmd = [
        str(binary),
        "--backend", args.backend,
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup),
        "--tries", str(args.tries),
        "--n", str(args.n),
        "--b", str(args.b),
        "--r", str(args.r),
        "--nbits", args.nbits,
        "--epsilon-bits", str(args.epsilon_bits),
        "--block-height", str(args.block_height),
        "--nonce-seed-height", str(args.nonce_seed_height),
        "--parent-mtp-seed-height", str(args.parent_mtp_seed_height),
        "--parent-mtp", str(args.parent_mtp),
        "--product-digest-height", str(args.product_digest_height),
        "--parallel", str(args.inner_parallel),
        "--async", args.async_prepare,
        "--gpu-inputs", args.gpu_inputs,
    ]
    maybe_add(cmd, "--batch-size", case.batch_size)
    maybe_add(cmd, "--pool-slots", case.pool_slots)
    maybe_add(cmd, "--solver-threads", case.solver_threads)
    if args.per_iteration:
        cmd.append("--per-iteration")
    return cmd


def build_gbt_command(
    binary: Path,
    args: argparse.Namespace,
    case: SweepCase,
    worker: WorkerSpec,
) -> list[str]:
    cmd = [
        str(binary),
        "--version", args.gbt_version,
        "--prev-hash", args.gbt_prev_hash,
        "--merkle-root", args.gbt_merkle_root,
        "--time", str(args.gbt_time),
        "--bits", args.nbits,
        "--seed-a", args.seed_a,
        "--seed-b", args.seed_b,
        "--block-height", str(args.block_height),
        "--parent-mtp", str(args.parent_mtp),
        "--matmul-n", str(args.n),
        "--matmul-b", str(args.b),
        "--matmul-r", str(args.r),
        "--epsilon-bits", str(args.epsilon_bits),
        "--nonce-start", str(worker.nonce_start),
        "--max-tries", str(args.tries),
        "--backend", args.backend,
    ]
    if args.gbt_max_seconds > 0:
        cmd.extend(["--max-seconds", str(args.gbt_max_seconds)])
    if args.share_target:
        cmd.extend(["--share-target", args.share_target])
    maybe_add(cmd, "--batch-size", case.batch_size)
    maybe_add(cmd, "--pool-slots", case.pool_slots)
    maybe_add(cmd, "--solver-threads", case.solver_threads)
    return cmd


def build_command(
    binary: Path,
    args: argparse.Namespace,
    case: SweepCase,
    worker: WorkerSpec,
) -> list[str]:
    if args.solver == "matmul-bench":
        return build_matmul_command(binary, args, case)
    return build_gbt_command(binary, args, case, worker)


def case_name(case: SweepCase) -> str:
    parts = [
        f"ppg{case.processes_per_gpu}",
        f"lookahead{case.lookahead}",
        f"batch{case.batch_size}",
        f"pool{case.pool_slots}",
        f"threads{case.solver_threads}",
        f"rep{case.repeat}",
    ]
    return "-".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in parts)


def extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def nested_number(data: dict[str, Any], keys: list[str]) -> float | None:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def metrics_from_json(data: dict[str, Any] | None) -> dict[str, float | int | str | None]:
    if data is None:
        return {"attempts": None, "elapsed_s": None, "nonces_per_sec": None}

    aggregate_nps = nested_number(data, ["measured_totals", "aggregate_nonces_per_sec"])
    attempts = nested_number(data, ["measured_totals", "total_attempts"])
    elapsed_s = nested_number(data, ["measured_totals", "total_elapsed_s"])
    if aggregate_nps is None:
        aggregate_nps = nested_number(data, ["nonces_per_sec", "mean"])
    if attempts is None:
        attempts = nested_number(data, ["total_attempts"])
    if elapsed_s is None:
        elapsed_s = nested_number(data, ["elapsed_s", "mean"])

    tries_used = nested_number(data, ["tries_used"])
    gbt_elapsed = nested_number(data, ["elapsed_s"])
    if attempts is None and tries_used is not None:
        attempts = tries_used
    if elapsed_s is None and gbt_elapsed is not None:
        elapsed_s = gbt_elapsed
    if aggregate_nps is None and attempts is not None and elapsed_s and elapsed_s > 0:
        aggregate_nps = attempts / elapsed_s

    return {
        "attempts": int(attempts) if attempts is not None else None,
        "elapsed_s": elapsed_s,
        "nonces_per_sec": aggregate_nps,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def launch_case(
    binary: Path,
    args: argparse.Namespace,
    case: SweepCase,
    gpus: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    workers = [
        WorkerSpec(
            index=index,
            gpu=gpu,
            local_index=local_index,
            nonce_start=args.nonce_start + (index * args.tries),
        )
        for index, (gpu, local_index) in enumerate(
            (gpu, local_index)
            for gpu in gpus
            for local_index in range(case.processes_per_gpu)
        )
    ]
    this_case_dir = run_dir / case_name(case)
    this_case_dir.mkdir(parents=True, exist_ok=True)

    process_records: list[dict[str, Any]] = []
    running: list[tuple[subprocess.Popen[None], Any, Any, dict[str, Any]]] = []
    case_started = time.monotonic()

    for worker in workers:
        env = worker_env(args, case, worker)
        cmd = build_command(binary, args, case, worker)
        stdout_path = this_case_dir / f"proc-{worker.index:02d}-gpu{worker.gpu}.stdout"
        stderr_path = this_case_dir / f"proc-{worker.index:02d}-gpu{worker.gpu}.stderr"
        record: dict[str, Any] = {
            "worker_index": worker.index,
            "gpu": worker.gpu,
            "gpu_local_process_index": worker.local_index,
            "nonce_start": worker.nonce_start,
            "command": cmd,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "env": {
                key: env[key]
                for key in [
                    "BTX_MATMUL_BACKEND",
                    "BTX_MATMUL_REQUIRE_BACKEND",
                    "BTX_MATMUL_GPU_INPUTS",
                    "BTX_MATMUL_CUDA_DEVICES",
                    "BTX_MATMUL_NONCE_SEED_LOOKAHEAD",
                    "BTX_MATMUL_SOLVE_BATCH_SIZE",
                    "BTX_MATMUL_CUDA_POOL_SLOTS",
                    "BTX_MATMUL_SOLVER_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                ]
                if key in env
            },
        }
        stdout_file = stdout_path.open("w", encoding="utf-8", errors="replace")
        stderr_file = stderr_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(cmd, env=env, stdout=stdout_file, stderr=stderr_file)
        running.append((proc, stdout_file, stderr_file, record))

    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds > 0 else None
    timed_out = False
    while True:
        if all(proc.poll() is not None for proc, _, _, _ in running):
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            for proc, _, _, _ in running:
                if proc.poll() is None:
                    proc.kill()
            break
        time.sleep(0.2)

    for proc, stdout_file, stderr_file, record in running:
        proc.wait()
        stdout_file.close()
        stderr_file.close()
        record["returncode"] = proc.returncode
        stdout_text = Path(record["stdout"]).read_text(encoding="utf-8", errors="replace")
        parsed = extract_json(stdout_text)
        record["parsed_json"] = parsed
        record["metrics"] = metrics_from_json(parsed)
        process_records.append(record)

    case_elapsed_s = time.monotonic() - case_started
    total_attempts = sum(
        int(record["metrics"]["attempts"])
        for record in process_records
        if record["metrics"].get("attempts") is not None
    )
    sum_reported_nps = sum(
        float(record["metrics"]["nonces_per_sec"])
        for record in process_records
        if record["metrics"].get("nonces_per_sec") is not None
    )
    aggregate_nps = total_attempts / case_elapsed_s if case_elapsed_s > 0 and total_attempts > 0 else None

    ok_return_codes = {0}
    if args.solver == "gbt-solve":
        ok_return_codes.add(2)
    ok = (not timed_out) and all(record["returncode"] in ok_return_codes for record in process_records)

    result = {
        "solver": args.solver,
        "case": case.__dict__,
        "case_dir": str(this_case_dir),
        "gpus": gpus,
        "total_processes": len(workers),
        "timed_out": timed_out,
        "ok": ok,
        "wall_elapsed_s": case_elapsed_s,
        "total_attempts": total_attempts,
        "aggregate_nonces_per_sec": aggregate_nps,
        "sum_reported_nonces_per_sec": sum_reported_nps,
        "processes": process_records,
    }
    write_json(this_case_dir / "case-summary.json", result)
    return result


def print_dry_run(binary: Path, args: argparse.Namespace, cases: list[SweepCase], gpus: list[str]) -> None:
    print(f"binary: {binary}")
    for case in cases:
        print(f"\ncase: {case_name(case)}")
        workers = [
            WorkerSpec(index=index, gpu=gpu, local_index=local_index, nonce_start=args.nonce_start + index * args.tries)
            for index, (gpu, local_index) in enumerate(
                (gpu, local_index)
                for gpu in gpus
                for local_index in range(case.processes_per_gpu)
            )
        ]
        for worker in workers:
            env = worker_env(args, case, worker)
            cmd = build_command(binary, args, case, worker)
            selected_env = {
                key: env[key]
                for key in sorted(env)
                if key.startswith("BTX_MATMUL_") or key == "CUDA_VISIBLE_DEVICES"
            }
            print(f"  worker {worker.index} gpu={worker.gpu} local={worker.local_index}")
            print("    env:", " ".join(f"{key}={value}" for key, value in selected_env.items()))
            print("    cmd:", " ".join(cmd))


def flatten_summary(result: dict[str, Any]) -> dict[str, Any]:
    case = result["case"]
    return {
        "solver": result["solver"],
        "processes_per_gpu": case["processes_per_gpu"],
        "gpus": ",".join(result["gpus"]),
        "total_processes": result["total_processes"],
        "lookahead": case["lookahead"],
        "batch_size": case["batch_size"],
        "pool_slots": case["pool_slots"],
        "solver_threads": case["solver_threads"],
        "repeat": case["repeat"],
        "ok": result["ok"],
        "timed_out": result["timed_out"],
        "wall_elapsed_s": result["wall_elapsed_s"],
        "total_attempts": result["total_attempts"],
        "aggregate_nonces_per_sec": result["aggregate_nonces_per_sec"],
        "sum_reported_nonces_per_sec": result["sum_reported_nonces_per_sec"],
        "case_dir": result["case_dir"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "ppg",
        "procs",
        "lookahead",
        "batch",
        "pool",
        "threads",
        "ok",
        "agg_nps",
        "attempts",
        "wall_s",
    ]
    print("\nSummary")
    print(" ".join(f"{header:>12}" for header in headers))
    for row in rows:
        agg = row["aggregate_nonces_per_sec"]
        agg_text = f"{agg:.2f}" if isinstance(agg, (int, float)) else "n/a"
        values = [
            row["processes_per_gpu"],
            row["total_processes"],
            row["lookahead"],
            row["batch_size"],
            row["pool_slots"],
            row["solver_threads"],
            row["ok"],
            agg_text,
            row["total_attempts"],
            f"{row['wall_elapsed_s']:.2f}",
        ]
        print(" ".join(f"{str(value):>12}" for value in values))


def build_cases(args: argparse.Namespace) -> list[SweepCase]:
    process_counts = parse_positive_csv(args.processes_per_gpu)
    lookahead_values = parse_csv_values(args.lookahead_values)
    batch_sizes = parse_csv_values(args.batch_sizes)
    pool_slots = parse_csv_values(args.pool_slots)
    solver_threads = parse_csv_values(args.solver_threads)
    cases: list[SweepCase] = []
    for repeat in range(1, args.repeats + 1):
        for ppg, lookahead, batch, pool, threads in itertools.product(
            process_counts,
            lookahead_values,
            batch_sizes,
            pool_slots,
            solver_threads,
        ):
            cases.append(SweepCase(ppg, lookahead, batch, pool, threads, repeat))
    return cases


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local BTX solver throughput with one or more processes per GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver", choices=["matmul-bench", "gbt-solve"], default="matmul-bench")
    parser.add_argument("--binary", help="Explicit solver binary path.")
    parser.add_argument("--build-dir", help="Build directory containing bin/<solver>.")
    parser.add_argument("--out-dir", default=str(repo_root() / ".btx-local-bench"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Case timeout; 0 disables.")

    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--no-require-backend", action="store_false", dest="require_backend")
    parser.set_defaults(require_backend=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA ordinals, or auto.")
    parser.add_argument("--cuda-visible-devices", action="store_true", help="Also set CUDA_VISIBLE_DEVICES per worker.")
    parser.add_argument("--processes-per-gpu", default="1,2")
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--lookahead-values", default="0")
    parser.add_argument("--batch-sizes", default="auto")
    parser.add_argument("--pool-slots", default="auto")
    parser.add_argument("--solver-threads", default="auto")
    parser.add_argument("--gpu-inputs", default="1")
    parser.add_argument("--async-prepare", default="1")

    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tries", type=int, default=4_194_304)
    parser.add_argument("--inner-parallel", type=int, default=1, help="btx-matmul-solve-bench --parallel.")
    parser.add_argument("--per-iteration", action="store_true")

    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--nbits", default="0x1e063c74")
    parser.add_argument("--epsilon-bits", type=int, default=18)
    parser.add_argument("--block-height", type=int, default=130_500)
    parser.add_argument("--nonce-seed-height", type=int, default=125_000)
    parser.add_argument("--parent-mtp-seed-height", type=int, default=130_500)
    parser.add_argument("--parent-mtp", type=int, default=1_780_000_000)
    parser.add_argument("--product-digest-height", type=int, default=61_000)
    parser.add_argument("--seed-a", default="6410ee507c58dca3d22f950385d38fdd5fba9dd2e424b2657a2410e92d23dc63")
    parser.add_argument("--seed-b", default="7f165f0361461f69e2442a31fec8c26d2d95928cae37cb1673cd14fbba25f03c")

    parser.add_argument("--nonce-start", type=int, default=1)
    parser.add_argument("--gbt-version", default="0x20000000")
    parser.add_argument("--gbt-prev-hash", default=HEX64_ZERO[:-2] + "11")
    parser.add_argument("--gbt-merkle-root", default=HEX64_ZERO[:-2] + "22")
    parser.add_argument("--gbt-time", type=int, default=1_780_000_000)
    parser.add_argument("--gbt-max-seconds", type=float, default=0.0)
    parser.add_argument("--share-target", default="")

    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.tries <= 0:
        parser.error("--tries must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be >= 0")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    binary = find_binary(args)
    gpus = detect_gpus(args.gpus)
    cases = build_cases(args)

    if args.dry_run:
        print_dry_run(binary, args, cases, gpus)
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir).expanduser() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": timestamp,
        "repo_root": str(repo_root()),
        "binary": str(binary),
        "argv": argv,
        "gpus": gpus,
        "case_count": len(cases),
    }
    write_json(run_dir / "manifest.json", manifest)

    results = []
    rows = []
    for case in cases:
        print(f"running {case_name(case)} ...", flush=True)
        result = launch_case(binary, args, case, gpus, run_dir)
        results.append(result)
        rows.append(flatten_summary(result))

    write_json(run_dir / "summary.json", {"manifest": manifest, "results": results})
    write_csv(run_dir / "summary.csv", rows)
    print_table(rows)
    print(f"\nArtifacts: {run_dir}")
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
