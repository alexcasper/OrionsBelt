"""Smoke tests that the orionsbelt src-layout package is importable.

These exist to catch packaging regressions (bad ``pyproject.toml``, missing
``__init__.py``, broken src-layout discovery) early and cheaply, independent
of any ML dependency. They intentionally avoid importing torch/transformers
or anything from the ``t-weights-fetch`` dependency set.
"""

import importlib
import pkgutil

import orionsbelt

EXPECTED_TOP_LEVEL_SUBMODULES = {"model", "engines", "partition", "quant"}

EXPECTED_ENGINE_SUBMODULES = {"npu", "gpu", "cpu"}


def test_top_level_package_importable():
    assert orionsbelt.__doc__ is not None
    assert hasattr(orionsbelt, "__all__")


def test_top_level_declares_expected_submodules():
    assert set(orionsbelt.__all__) == EXPECTED_TOP_LEVEL_SUBMODULES


def test_top_level_submodules_import_and_match_all():
    discovered = {info.name for info in pkgutil.iter_modules(orionsbelt.__path__)}

    # Every name advertised in __all__ must actually exist as a submodule ...
    assert discovered >= EXPECTED_TOP_LEVEL_SUBMODULES

    # ... and must actually import cleanly.
    for name in EXPECTED_TOP_LEVEL_SUBMODULES:
        module = importlib.import_module(f"orionsbelt.{name}")
        assert module.__doc__, f"orionsbelt.{name} is missing a module docstring"


def test_engines_exposes_per_accelerator_backends():
    engines = importlib.import_module("orionsbelt.engines")
    discovered = {info.name for info in pkgutil.iter_modules(engines.__path__)}

    assert discovered >= EXPECTED_ENGINE_SUBMODULES

    for name in EXPECTED_ENGINE_SUBMODULES:
        module = importlib.import_module(f"orionsbelt.engines.{name}")
        assert module.__doc__, f"orionsbelt.engines.{name} is missing a module docstring"
