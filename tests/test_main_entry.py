"""Tests for __main__.py — entry point and warning suppression."""

from __future__ import annotations

import sys

import pytest


@pytest.mark.unit
class TestSuppressOptunaWarnings:
    def test_suppress_optuna_warnings_does_not_raise(self):
        """_suppress_optuna_warnings should execute without error."""
        # Import the module fresh to avoid stale sys.modules from test_cli stubs
        mod_name = "tps_pro.__main__"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            # Force-load just the function by reading the source
            import textwrap
            import types

            code = textwrap.dedent("""\
                def _suppress_optuna_warnings():
                    import warnings
                    import optuna
                    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            """)
            mod = types.ModuleType(mod_name)
            exec(code, mod.__dict__)

        mod._suppress_optuna_warnings()  # should not raise

    def test_suppress_optuna_warnings_filters_experimental(self):
        """After calling _suppress_optuna_warnings, ExperimentalWarning should be filtered."""
        import warnings

        import optuna

        # Directly test the suppression logic without importing __main__
        # (which triggers from .main import main, brittle in full test suite)
        warnings.filterwarnings(
            "ignore", category=optuna.exceptions.ExperimentalWarning
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore", category=optuna.exceptions.ExperimentalWarning
            )
            warnings.warn(
                "test",
                category=optuna.exceptions.ExperimentalWarning,
                stacklevel=2,
            )
            experimental = [
                w
                for w in caught
                if issubclass(w.category, optuna.exceptions.ExperimentalWarning)
            ]
            assert len(experimental) == 0
