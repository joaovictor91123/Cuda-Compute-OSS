"""Invalid CLI arguments must be reported cleanly (exit code 2 + an ``error:``
line on stderr), never surfaced as an uncaught traceback.

Both ``matmul`` and ``strategy`` validate ``--n`` and build/validate their
``Config`` *before* any device work, so every case here is rejected on CPU with
no GPU/PyTorch present -- which is exactly how they run in PR CI.
"""
from __future__ import annotations

import pytest

from matmul import cli as matmul_cli
from strategy import cli as strategy_cli

BAD_ARGS = [
    ["--vram-fraction", "1.5", "--n", "8"],   # vram_fraction > 0.95
    ["--vram-fraction", "0", "--n", "8"],     # vram_fraction <= 0
    ["--n", "0"],                             # non-positive n
    ["--n", "-4"],                            # negative n
]


@pytest.mark.parametrize("main", [matmul_cli.main, strategy_cli.main],
                         ids=["matmul", "strategy"])
@pytest.mark.parametrize("argv", BAD_ARGS, ids=lambda a: " ".join(a))
def test_bad_args_exit_cleanly(main, argv, capsys):
    rc = main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


# --data-rank is strategy-only (matmul has no such flag), so it can't share
# BAD_ARGS above, which is parametrized across both CLIs.
STRATEGY_BAD_DATA_RANK_ARGS = [
    ["--n", "8", "--data-rank", "0"],    # non-positive: a rank-0 "benchmark"
                                          # isn't a meaningful input, same as --n 0
    ["--n", "8", "--data-rank", "-3"],   # negative: previously an uncaught
                                          # traceback from _fill_lowrank's
                                          # 1/sqrt(rank) and negative-size rng draw
]


@pytest.mark.parametrize("argv", STRATEGY_BAD_DATA_RANK_ARGS, ids=lambda a: " ".join(a))
def test_bad_data_rank_exits_cleanly(argv, capsys):
    rc = strategy_cli.main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


def test_positive_data_rank_is_unaffected(capsys):
    # --data-rank 1 (smallest valid rank) must not be rejected by validation --
    # this guards against the check being off-by-one. This test runs without a
    # GPU, so a rc==2 here may legitimately come from the "no GPU" path (this
    # module computes on GPU only); it must never come from a --data-rank
    # complaint.
    rc = strategy_cli.main(["--n", "8", "--data-rank", "1", "--quiet"])
    if rc == 2:
        assert "--data-rank" not in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
