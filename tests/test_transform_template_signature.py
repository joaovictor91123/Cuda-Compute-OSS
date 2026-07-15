"""Pin the contributor transform template to the real basis() contract.

strategy/examples/transform_template.py is the copy-paste starting point for a
new Transform, so whatever signature it shows is the signature contributors
ship. `multiply_subspace` forwards the VRAM budget only to a basis() that
declares `frac`:

    if "frac" in inspect.signature(transform.basis).parameters:
        Q = transform.basis(n, m, backend, cdt, A=A, B=B, frac=frac)
    else:
        Q = transform.basis(n, m, backend, cdt, A=A, B=B)

The template had been left on the pre-#211 signature (no `frac`), so every
transform copied from it lands in the `else` branch and silently streams its
basis at the 0.3 default instead of Config.vram_fraction -- the exact bug #211
fixed for rsvd, re-introduced by each new contribution.

These tests keep the template's parameters in step with the Transform base class
(and rsvd), so the template cannot drift behind the interface again. Pure
introspection; no GPU needed.

Run:  python tests/test_transform_template_signature.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy", "examples"))

from strategy.transforms import RandomizedSVDTransform, Transform

import transform_template


def _params(fn) -> list:
    return list(inspect.signature(fn).parameters)


def test_template_basis_declares_frac():
    """Without `frac`, multiply_subspace takes the legacy branch and the copied
    transform silently ignores --vram-fraction."""
    assert "frac" in _params(transform_template.MyTransform.basis)


def test_template_basis_matches_the_base_class_contract():
    """The template is what contributors ship -- keep its parameters identical
    to the interface it implements."""
    assert _params(transform_template.MyTransform.basis) == _params(Transform.basis)


def test_builtin_rsvd_agrees_with_the_template():
    """Sanity: the shipped data-dependent transform declares the same knobs."""
    assert _params(RandomizedSVDTransform.basis) == _params(transform_template.MyTransform.basis)


def test_template_documents_frac():
    doc = inspect.getdoc(transform_template.MyTransform.basis) or ""
    assert "frac" in doc, "template's basis() docstring must explain frac"


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
