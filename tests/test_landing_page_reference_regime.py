"""Pin the landing page's reference-regime claims to the code (issue: stale N).

`index/index.html` advertises CCO's reference regime and a copy-pasteable
quickstart command. Those must track the real default the engines ship with --
`strategy`/`matmul` `--n` (8192), which README.md and BENCHMARKS.md also state as
"Reference setup: 8192 x 8192". The page had been left at the pre-8192 default of
N=12,000 (and its KPI showed 2*12000**3 = 3.5 TFLOP), so a visitor's first
command and headline number disagreed with every other surface in the repo.

These tests re-derive the page's claims from the CLI default itself, so the
landing page cannot silently drift away from the shipped default again.
Pure parsing + arithmetic; no GPU needed.

Run:  python tests/test_landing_page_reference_regime.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.cli import build_parser as strategy_parser
from matmul.cli import build_parser as matmul_parser

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAGE = os.path.join(_ROOT, "index", "index.html")


def _page() -> str:
    with open(_PAGE, encoding="utf-8") as fh:
        return fh.read()


def _default_n() -> int:
    n = strategy_parser().parse_args([]).n
    assert n == matmul_parser().parse_args([]).n, "engines disagree on default --n"
    return n


def test_quickstart_command_uses_the_shipped_default_n():
    """The copy-pasteable `python -m eval --n ...` on the page must be the default."""
    shown = re.findall(r"python -m eval --n (\d+)", _page())
    assert shown, "no `python -m eval --n <N>` quickstart found on the landing page"
    for value in shown:
        assert int(value) == _default_n()


def test_reference_regime_card_matches_the_default_n():
    """The 'CCO's reference regime' KPI card must name the same N."""
    m = re.search(r"N = ([\d,]+)\s*&mdash;|N = ([\d,]+)\s*—\s*CCO's reference regime", _page())
    assert m, "no 'CCO's reference regime' card found"
    shown = int((m.group(1) or m.group(2)).replace(",", ""))
    assert shown == _default_n()


def test_reference_regime_flop_kpi_is_2n3_at_the_default_n():
    """The KPI beside the reference regime is 2*N**3; check it in decimal units
    (the same convention the 1,000 -> 2 GFLOP and 128,000 -> 4.2 PFLOP cards use)."""
    n = _default_n()
    expected_tflop = round(2.0 * n**3 / 1e12, 1)
    assert expected_tflop == 1.1, f"sanity: 2*{n}**3 should be 1.1 TFLOP, got {expected_tflop}"
    assert f"{expected_tflop} TFLOP" in _page()
    # the pre-8192 figure (2*12000**3) must be gone
    assert "3.5 TFLOP" not in _page()


def test_no_stale_12000_reference_remains():
    page = _page()
    assert "12000" not in page and "12,000" not in page


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
