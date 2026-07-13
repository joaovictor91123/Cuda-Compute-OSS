"""Sequential GPU batch runner for queued PRs.

This is the Phase 2 bridge between the always-on PR bot and the later live GPU
scorer. It consumes ``dashboard/data.json`` (written by ``eval.pr_bot``), takes
queued PRs in oldest-first order, and either prints or executes the exact steps
for a maintainer-controlled GPU window.

Default mode is dry-run. Use ``--run`` only on a disposable GPU machine or a
properly isolated self-hosted runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from . import tracks

DEFAULT_QUEUE = "dashboard/data.json"
DEFAULT_WORKDIR = "_gpu_batch_work"
DEFAULT_RESULTS_DIR = "gpu-results"
MOCK_GPU_NAME = "RTX 5090 (mock)"
DEFAULT_RUNS = 5          # fresh unseen seeds per PR; verdict is worst-case over them


@dataclass(frozen=True)
class QueueItem:
    pr: int
    title: str
    author: str
    head_sha: str
    position: int | None = None
    url: str = ""
    track: str | None = None      # declared track -> pinned regime (eval.tracks)


def spec_for_track(spec: EvalSpec, track: str | None) -> "EvalSpec":
    """Override the regime knobs with the declared track's PINNED regime, so a PR
    is scored at its track's fixed (fill, rank, M) — not knobs it chose. Unknown
    or unspecified track => unchanged (falls back to the full-rank reference)."""
    if not track or track not in tracks.TRACKS:
        return spec
    ts = tracks.TRACKS[track]
    return replace(
        spec,
        fill=ts.fill,
        data_rank=ts.data_rank,
        rank_m=ts.rank_m if ts.rank_m is not None else spec.rank_m,
    )


def aggregate_runs(runs: list[dict]) -> dict:
    """Combine K per-seed eval outputs into ONE worst-case verdict per transform.

    A transform is admitted only if it dominates exact on EVERY run (no lucky
    seed): accuracy=min, latency=max, VRAM=max across the K runs; improvement
    requires all runs dominant; gated if any run gated; recorded score is the
    min (worst) across runs, else 0. Pure — no I/O, so it's fully unit-tested."""
    if not runs:
        raise ValueError("aggregate_runs needs at least one run")
    base = dict(runs[0])                                   # config/exact/complexity from run 0
    names = set(runs[0].get("transforms", {}))
    for r in runs[1:]:
        names &= set(r.get("transforms", {}))
    agg: dict = {}
    for name in names:
        cells = [r["transforms"][name] for r in runs]
        dominant = all(c.get("improvement") for c in cells)
        agg[name] = {
            "accuracy": min(c["accuracy"] for c in cells),
            "latency_s": max(c["latency_s"] for c in cells),
            "peak_vram_bytes": max(c["peak_vram_bytes"] for c in cells),
            "peak_vram_mib": max(c["peak_vram_mib"] for c in cells),
            "flop_ratio_vs_exact": min(c["flop_ratio_vs_exact"] for c in cells),
            "faster_than_exact": all(c.get("faster_than_exact") for c in cells),
            "less_vram_than_exact": all(c.get("less_vram_than_exact") for c in cells),
            "fewer_flops_than_exact": all(c.get("fewer_flops_than_exact") for c in cells),
            "gated": any(c.get("gated") for c in cells),
            "improvement": dominant,
            "score": min(c.get("score", 0.0) for c in cells) if dominant else 0.0,
            "runs": len(cells),
            "seeds": [r.get("config", {}).get("seed") for r in runs],
        }
    ranking = sorted(agg, key=lambda k: agg[k]["score"], reverse=True)
    base["transforms"] = agg
    base["ranking"] = ranking
    base["best"] = ranking[0] if ranking else None
    base["aggregation"] = {"runs": len(runs), "rule": "worst-case over fresh unseen seeds"}
    return base


@dataclass(frozen=True)
class EvalSpec:
    n: int = 8192
    pairs: int = 3
    dtype: str = "fp32"
    rank_m: int | None = None
    fill: str = "random"
    data_rank: int | None = None
    transforms: str | None = None
    seed: int | None = None
    device: int = 0


def load_queue(path: str | Path) -> list[QueueItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for raw in data.get("queue", []):
        items.append(
            QueueItem(
                pr=int(raw["pr"]),
                title=raw.get("title", ""),
                author=raw.get("author", ""),
                head_sha=raw.get("head_sha", ""),
                position=raw.get("position"),
                url=raw.get("url", ""),
                track=raw.get("track"),
            )
        )
    return sorted(items, key=lambda item: item.position or item.pr)


def select_batch(queue: list[QueueItem], limit: int | None) -> list[QueueItem]:
    if limit is None or limit <= 0:
        return queue
    return queue[:limit]


def eval_args(spec: EvalSpec) -> list[str]:
    args = [
        "uv", "run", "python", "-m", "eval",
        "--n", str(spec.n),
        "--pairs", str(spec.pairs),
        "--dtype", spec.dtype,
        "--fill", spec.fill,
        "--device", str(spec.device),
        "--json",
    ]
    if spec.rank_m is not None:
        args += ["--rank-m", str(spec.rank_m)]
    if spec.data_rank is not None:
        args += ["--data-rank", str(spec.data_rank)]
    if spec.transforms:
        args += ["--transforms", spec.transforms]
    if spec.seed is not None:
        args += ["--seed", str(spec.seed)]
    return args


def result_path(item: QueueItem, results_dir: str | Path) -> Path:
    return Path(results_dir) / f"pr-{item.pr}-{item.head_sha[:12] or 'unknown'}.json"


def _mock_seed(item: QueueItem, spec: EvalSpec) -> int:
    if spec.seed is not None:
        return spec.seed
    material = f"{item.pr}:{item.head_sha}:{spec.n}:{spec.pairs}:{spec.fill}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def mock_result(item: QueueItem, spec: EvalSpec) -> dict:
    """Produce an evaluate()-shaped result without requiring GPU hardware."""
    transform = (spec.transforms.split(",")[0].strip() if spec.transforms else "mock_transform")
    seed = _mock_seed(item, spec)
    score = 10.0 + (item.pr % 7) / 10.0
    result = {
        "accuracy": 0.94,
        "rel_frobenius_error": 0.06,
        "latency_s": 0.021 + (item.pr % 3) * 0.002,
        "peak_vram_bytes": 1_610_612_736,
        "peak_vram_mib": 1536.0,
        "flop_ratio_vs_exact": 2.5,
        "faster_than_exact": True,
        "less_vram_than_exact": True,
        "fewer_flops_than_exact": True,
        "gated": False,
        "improvement": True,
        "perf_score": score,
        "score": score,
    }
    return {
        "pr": item.pr,
        "title": item.title,
        "author": item.author,
        "head_sha": item.head_sha,
        "url": item.url,
        "mock": True,
        "eval": {
            "config": {
                "n": spec.n,
                "pairs": spec.pairs,
                "dtype": spec.dtype,
                "rank_m": spec.rank_m,
                "fill": spec.fill,
                "accuracy_floor": 0.8,
                "vram_unit": "gib",
                "device": MOCK_GPU_NAME,
                "seed": seed,
            },
            "complexity": {"normal": "O(N^3)", "smart": "O(N^2 * M)"},
            "exact": {
                "latency_s": 0.052,
                "peak_vram_bytes": 4_294_967_296,
                "peak_vram_mib": 4096.0,
            },
            "transforms": {transform: result},
            "ranking": [transform],
            "best": transform,
        },
    }


def wrap_result(item: QueueItem, eval_output: str, *, mock: bool = False) -> dict:
    return {
        "pr": item.pr,
        "title": item.title,
        "author": item.author,
        "head_sha": item.head_sha,
        "url": item.url,
        "mock": mock,
        "eval": json.loads(eval_output),
    }


def plan_item(
    item: QueueItem,
    *,
    repo: str,
    workdir: str | Path,
    results_dir: str | Path,
    spec: EvalSpec,
) -> list[str]:
    checkout = Path(workdir) / f"pr-{item.pr}"
    result = (Path.cwd() / result_path(item, results_dir))
    return [
        f"gh repo clone {repo} {checkout}",
        f"cd {checkout} && gh pr checkout {item.pr}",
        f"cd {checkout} && test \"$(git rev-parse HEAD)\" = \"{item.head_sha}\"",
        f"cd {checkout} && uv sync --extra test --extra gpu",
        f"cd {checkout} && uv run --extra test python -m py_compile $(find matmul strategy eval tests examples -name '*.py')",
        f"cd {checkout} && uv run --extra test python -m pytest tests/ strategy/tests/ eval/tests/ -v",
        f"cd {checkout} && uv run python -m strategy.smoke",
        "cd "
        + str(checkout)
        + " && "
        + " ".join(eval_args(spec))
        + " > "
        + str(result),
    ]


def _run(cmd: list[str] | str, *, cwd: str | Path | None = None, capture: bool = False):
    return subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        text=True,
        capture_output=capture,
        check=True,
    )


def _rebase_onto_main(checkout: Path) -> bool:
    """Rebase the checked-out PR onto origin/main. Return True on success; on a
    conflict, abort cleanly and return False (the PR must be scored against the
    CURRENT frontier + shared code, never its stale branch)."""
    _run(["git", "fetch", "origin", "main"], cwd=checkout)
    r = subprocess.run(["git", "rebase", "origin/main"], cwd=checkout,
                       text=True, capture_output=True)
    if r.returncode != 0:
        subprocess.run(["git", "rebase", "--abort"], cwd=checkout,
                       text=True, capture_output=True)
        return False
    return True


def run_item(
    item: QueueItem,
    *,
    repo: str,
    workdir: str | Path,
    results_dir: str | Path,
    spec: EvalSpec,
    clean: bool = False,
    mock: bool = False,
    runs: int = DEFAULT_RUNS,
    sweep: str | None = None,
) -> Path:
    """Score one queued PR: rebase onto main, run the declared track's PINNED
    regime over ``runs`` fresh unseen seeds, and record the WORST-CASE verdict.
    Returns the JSON result path."""
    workdir = Path(workdir)
    results_dir = Path(results_dir)
    checkout = workdir / f"pr-{item.pr}"
    result = result_path(item, results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Pin the regime to the declared track (ignore any knobs the PR chose).
    spec = spec_for_track(spec, item.track)

    if mock:
        result.write_text(json.dumps(mock_result(item, spec), indent=2) + "\n",
                          encoding="utf-8")
        return result

    if checkout.exists():
        if not clean:
            raise FileExistsError(f"{checkout} already exists; pass --clean to replace it")
        shutil.rmtree(checkout)

    workdir.mkdir(parents=True, exist_ok=True)

    _run(["gh", "repo", "clone", repo, str(checkout)])
    _run(["gh", "pr", "checkout", str(item.pr)], cwd=checkout)
    actual_sha = _run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).stdout.strip()
    if item.head_sha and actual_sha != item.head_sha:
        raise RuntimeError(
            f"PR #{item.pr} checked out {actual_sha}, expected queued SHA {item.head_sha}"
        )

    # Score the MERGED state, not the branch: rebase onto current main so the
    # frontier transform (rsvd) and shared code are current. Conflict => skip.
    if not _rebase_onto_main(checkout):
        result.write_text(json.dumps({
            "pr": item.pr, "title": item.title, "author": item.author,
            "head_sha": item.head_sha, "url": item.url, "mock": False,
            "state": "needs_rebase",
            "detail": "conflicts with main; contributor must rebase before scoring",
        }, indent=2) + "\n", encoding="utf-8")
        return result

    _run(["uv", "sync", "--extra", "test", "--extra", "gpu"], cwd=checkout)
    _run("uv run --extra test python -m py_compile $(find matmul strategy eval tests examples -name '*.py')",
         cwd=checkout)
    _run(["uv", "run", "--extra", "test", "python", "-m", "pytest",
          "tests/", "strategy/tests/", "eval/tests/", "-v"], cwd=checkout)
    _run(["uv", "run", "python", "-m", "strategy.smoke"], cwd=checkout)

    # K fresh unseen seeds (spec.seed stays None so each run draws its own), then
    # collapse to the worst case -- a real win survives every seed.
    outputs = []
    for _ in range(max(1, runs)):
        completed = _run(eval_args(spec), cwd=checkout, capture=True)
        outputs.append(json.loads(completed.stdout))
    aggregate = aggregate_runs(outputs)

    # Empirical scaling fit (the sub-cubic proof), once.
    if sweep:
        sw = _run(eval_args(replace(spec, seed=None)) + ["--sweep", sweep],
                  cwd=checkout, capture=True)
        sweep_out = json.loads(sw.stdout).get("scaling")
        if sweep_out:
            aggregate["scaling"] = sweep_out

    wrapped = {
        "pr": item.pr, "title": item.title, "author": item.author,
        "head_sha": item.head_sha, "url": item.url, "mock": False,
        "track": item.track, "eval": aggregate,
    }
    result.write_text(json.dumps(wrapped, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.gpu_batch",
        description="Run or print the next sequential GPU evaluation batch.",
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--repo", default="zeokin/Cuda-Compute-OSS")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=1,
                        help="number of queued PRs to evaluate; <=0 means all")
    parser.add_argument("--run", action="store_true",
                        help="execute the batch. Omit for a dry-run plan.")
    parser.add_argument("--mock", action="store_true",
                        help="with --run, write mock RTX 5090 result JSON without gh/GPU")
    parser.add_argument("--clean", action="store_true",
                        help="replace existing per-PR checkout directories")
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp16", "fp32", "fp64"), default="fp32")
    parser.add_argument("--rank-m", type=int, default=None)
    parser.add_argument("--fill", choices=("random", "lowrank", "decaying-spectrum", "iota"),
                        default="random")
    parser.add_argument("--data-rank", type=int, default=None)
    parser.add_argument("--transforms", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="omit for fresh unseen inputs; pass only to reproduce a run")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="fresh unseen seeds per PR; verdict is worst-case over them")
    parser.add_argument("--sweep", default=None,
                        help="comma-separated N sizes for the scaling fit (e.g. 2048,4096,8192)")
    args = parser.parse_args(argv)

    spec = EvalSpec(
        n=args.n,
        pairs=args.pairs,
        dtype=args.dtype,
        rank_m=args.rank_m,
        fill=args.fill,
        data_rank=args.data_rank,
        transforms=args.transforms,
        seed=args.seed,
        device=args.device,
    )
    batch = select_batch(load_queue(args.queue), args.limit)
    if not batch:
        print("No queued PRs found.")
        return 0

    for item in batch:
        print(f"PR #{item.pr} ({item.author}): {item.title}")
        if args.run:
            result = run_item(
                item,
                repo=args.repo,
                workdir=args.workdir,
                results_dir=args.results_dir,
                spec=spec,
                clean=args.clean,
                mock=args.mock,
                runs=args.runs,
                sweep=args.sweep,
            )
            print(f"  wrote {result}")
        else:
            for command in plan_item(
                item,
                repo=args.repo,
                workdir=args.workdir,
                results_dir=args.results_dir,
                spec=spec,
            ):
                print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
