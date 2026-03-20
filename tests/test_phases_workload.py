"""Tests for phases/workload.py — workload simulation and context sweep."""

from __future__ import annotations

import inspect

import pytest

from tps_pro.phases.workload import (
    phase_context_sweep,
    phase_workload_sim,
)


@pytest.mark.unit
class TestPhaseWorkload:
    def test_phase_workload_sim_is_callable(self):
        """phase_workload_sim should be a callable function."""
        assert callable(phase_workload_sim)

    def test_phase_context_sweep_is_callable(self):
        """phase_context_sweep should be a callable function."""
        assert callable(phase_context_sweep)

    def test_phase_workload_sim_signature(self):
        """phase_workload_sim should accept ctx and base_config."""
        sig = inspect.signature(phase_workload_sim)
        params = list(sig.parameters.keys())
        assert "ctx" in params
        assert "base_config" in params
