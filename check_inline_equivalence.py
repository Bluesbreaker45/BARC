"""
Check that inlined problem files behave identically to their original counterparts.

For each problem present in both `--original` and `--inline` directories:
  1. Run `generate_input()` in both versions with the same RNG seed; verify equality.
  2. Feed the shared input into `main()` in both versions (again with the same seed);
     verify equality.
  3. Repeat for several seeds.

Each version is executed in a fresh subprocess so that `from common import *`
resolves to the correct `common.py` (original vs. inlined) via sys.path.

Usage:
    # Compare seeds vs seeds-inline (smoke test on 10 problems, 3 trials each):
    python check_inline_equivalence.py \
        --original seeds --original-common seeds \
        --inline   seeds-inline --inline-common   seeds-inline \
        --limit 10

    # Compare synthetic_problems vs synthetic_problems-inline:
    python check_inline_equivalence.py

    # Check a single problem:
    python check_inline_equivalence.py --problem 00339623fade9c57.py
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import traceback
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent


# ──────────────────────── subprocess worker ────────────────────────


def worker_main():
    """Run in a fresh subprocess: read JSON instruction from stdin, write JSON result to stdout.

    Instruction fields:
        file: path to the .py problem file to load
        lib_dir: directory containing common.py (added to sys.path FIRST)
        seed: int RNG seed applied before each call
        call: "generate_input" | "main"
        input_grid: 2D list (only for call="main")
    """
    data = json.loads(sys.stdin.read())
    lib_dir = data["lib_dir"]
    file_path = data["file"]
    seed = data["seed"]
    call = data["call"]

    # Put lib_dir FIRST so `from common import *` picks the right common.py.
    sys.path.insert(0, lib_dir)

    import random

    import numpy as np

    with open(file_path) as f:
        source = f.read()

    result = {}
    globals_dict: dict = {"__name__": "__barc_problem__"}

    try:
        exec(compile(source, file_path, "exec"), globals_dict)
    except Exception as e:
        result["error"] = f"exec failed: {e}\n{traceback.format_exc()}"
        sys.stdout.write(json.dumps(result))
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        if call == "generate_input":
            if "generate_input" not in globals_dict:
                result["error"] = "no generate_input"
            else:
                out = globals_dict["generate_input"]()
                result["grid"] = _grid_to_list(out)
        elif call == "main":
            if "main" not in globals_dict:
                result["error"] = "no main"
            else:
                input_grid = np.array(data["input_grid"])
                out = globals_dict["main"](input_grid)
                result["grid"] = _grid_to_list(out)
        else:
            result["error"] = f"unknown call: {call}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    sys.stdout.write(json.dumps(result))


def _grid_to_list(g):
    """Normalize a grid return value to a JSON-serializable list-of-lists, or None."""
    import numpy as np

    if isinstance(g, np.ndarray):
        return g.tolist()
    if isinstance(g, list):
        return g
    return None


# ─────────────────────── driver (runs subprocesses) ───────────────────────


def run_worker(file_path, lib_dir, seed, call, input_grid=None, timeout=10):
    instruction = {
        "file": str(file_path),
        "lib_dir": str(lib_dir),
        "seed": seed,
        "call": call,
    }
    if input_grid is not None:
        instruction["input_grid"] = input_grid

    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--worker"],
            input=json.dumps(instruction),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}

    if proc.returncode != 0:
        return {"error": f"subprocess rc={proc.returncode}: {proc.stderr.strip()[:500]}"}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {
            "error": f"bad worker json: {e} | stdout[:300]={proc.stdout[:300]!r} | stderr[:300]={proc.stderr[:300]!r}"
        }


def compare_problem(
    problem_name,
    orig_file_dir,
    inline_file_dir,
    orig_lib_dir,
    inline_lib_dir,
    num_trials,
    base_seed,
    timeout,
):
    orig_file = orig_file_dir / problem_name
    inline_file = inline_file_dir / problem_name

    if not orig_file.exists() or not inline_file.exists():
        return {
            "status": "missing",
            "detail": f"orig exists={orig_file.exists()} inline exists={inline_file.exists()}",
        }

    mismatches = []

    for trial in range(num_trials):
        seed = base_seed + trial

        # Step 1: generate_input() in both versions with the same seed.
        orig_gen = run_worker(orig_file, orig_lib_dir, seed, "generate_input", timeout=timeout)
        inline_gen = run_worker(inline_file, inline_lib_dir, seed, "generate_input", timeout=timeout)

        if "error" in orig_gen:
            mismatches.append(f"seed={seed} original generate_input error: {orig_gen['error']}")
            continue
        if "error" in inline_gen:
            mismatches.append(f"seed={seed} inlined  generate_input error: {inline_gen['error']}")
            continue

        if orig_gen.get("grid") != inline_gen.get("grid"):
            mismatches.append(
                f"seed={seed} generate_input MISMATCH: "
                f"orig shape={_shape(orig_gen.get('grid'))} "
                f"inline shape={_shape(inline_gen.get('grid'))}"
            )
            continue

        shared_input = orig_gen.get("grid")
        if shared_input is None:
            mismatches.append(f"seed={seed} generate_input returned None/non-grid")
            continue

        # Step 2: main(shared_input) in both versions with the same seed.
        orig_out = run_worker(
            orig_file, orig_lib_dir, seed, "main", input_grid=shared_input, timeout=timeout
        )
        inline_out = run_worker(
            inline_file, inline_lib_dir, seed, "main", input_grid=shared_input, timeout=timeout
        )

        if "error" in orig_out:
            mismatches.append(f"seed={seed} original main error: {orig_out['error']}")
            continue
        if "error" in inline_out:
            mismatches.append(f"seed={seed} inlined  main error: {inline_out['error']}")
            continue

        if orig_out.get("grid") != inline_out.get("grid"):
            mismatches.append(
                f"seed={seed} main MISMATCH: "
                f"orig shape={_shape(orig_out.get('grid'))} "
                f"inline shape={_shape(inline_out.get('grid'))}"
            )

    return {"status": "ok" if not mismatches else "mismatch", "mismatches": mismatches}


def _shape(grid):
    if grid is None:
        return None
    if not grid or not isinstance(grid, list):
        return "?"
    return (len(grid), len(grid[0]) if isinstance(grid[0], list) else "?")


# ───────────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--original",
        default="synthetic_problems",
        help="Dir containing original problem .py files",
    )
    parser.add_argument(
        "--inline",
        default="synthetic_problems-inline",
        help="Dir containing inlined problem .py files",
    )
    parser.add_argument(
        "--original-common",
        default="seeds",
        help="Dir containing the original common.py (added to sys.path for the original version)",
    )
    parser.add_argument(
        "--inline-common",
        default="seeds-inline",
        help="Dir containing the inlined common.py (added to sys.path for the inlined version)",
    )

    parser.add_argument("--trials", type=int, default=3, help="Seeds to try per problem")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-subprocess timeout (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to check")
    parser.add_argument("--problem", type=str, default=None, help="Only check this filename, e.g. 00d62c1b.py")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of problems to compare in parallel (default: min(40, number of problems))",
    )
    parser.add_argument("--verbose", action="store_true", help="Print OK lines too")

    args = parser.parse_args()

    if args.worker:
        worker_main()
        return

    orig_dir = (REPO_ROOT / args.original).resolve()
    inline_dir = (REPO_ROOT / args.inline).resolve()
    orig_lib = (REPO_ROOT / args.original_common).resolve()
    inline_lib = (REPO_ROOT / args.inline_common).resolve()

    for d, label in [(orig_dir, "--original"), (inline_dir, "--inline"),
                     (orig_lib, "--original-common"), (inline_lib, "--inline-common")]:
        if not d.is_dir():
            print(f"error: {label}={d} is not a directory", file=sys.stderr)
            sys.exit(2)

    if args.problem:
        problems = [args.problem]
    else:
        skip_prefixes = ("common", "inline", "count_common_calls", "check_inline_equivalence")
        problems = sorted(
            f.name
            for f in orig_dir.glob("*.py")
            if not f.name.startswith(skip_prefixes)
        )

    if args.limit:
        problems = problems[: args.limit]

    if args.jobs is not None and args.jobs < 1:
        print("error: --jobs must be >= 1", file=sys.stderr)
        sys.exit(2)

    jobs = args.jobs if args.jobs is not None else min(40, max(1, len(problems)))

    print(f"Checking {len(problems)} problems")
    print(f"  original: {orig_dir}  (common from {orig_lib})")
    print(f"  inlined:  {inline_dir}  (common from {inline_lib})")
    print(f"  trials per problem: {args.trials}")
    print(f"  parallel jobs: {jobs}")
    print()

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        for name in problems:
            futures[name] = executor.submit(
                compare_problem,
                name,
                orig_dir,
                inline_dir,
                orig_lib,
                inline_lib,
                args.trials,
                args.base_seed,
                args.timeout,
            )

    ok = mismatch = missing = 0
    for i, name in enumerate(problems, 1):
        try:
            result = futures[name].result()
        except Exception as e:
            result = {"status": "error", "detail": f"{type(e).__name__}: {e}"}

        status = result["status"]
        if status == "ok":
            ok += 1
            if args.verbose:
                print(f"[{i}/{len(problems)}] {name}: OK")
        elif status == "mismatch":
            mismatch += 1
            print(f"[{i}/{len(problems)}] {name}: MISMATCH ({len(result['mismatches'])})")
            for msg in result["mismatches"]:
                print(f"    {msg}")
        else:
            missing += 1
            print(f"[{i}/{len(problems)}] {name}: {status} ({result.get('detail', '')})")

    print()
    print(f"Summary: {ok} OK, {mismatch} mismatch, {missing} missing/error")
    sys.exit(0 if mismatch == 0 and missing == 0 else 1)


if __name__ == "__main__":
    main()
