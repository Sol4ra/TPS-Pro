"""Behavioral tests for the errors.py exception hierarchy.

Verifies construction, inheritance chains, and string representation
for all custom exception classes.
"""

from __future__ import annotations

import pytest

from tps_pro.errors import (
    BaselineFailure,
    BenchOOMError,
    OptimizerError,
    ServerError,
)

# ===================================================================
# OptimizerError (base)
# ===================================================================


@pytest.mark.unit
class TestOptimizerError:
    def test_is_exception_subclass(self):
        assert issubclass(OptimizerError, Exception)

    def test_construction_with_message(self):
        err = OptimizerError("something went wrong")
        assert str(err) == "something went wrong"

    def test_construction_without_message(self):
        err = OptimizerError()
        assert str(err) == ""

    def test_can_be_raised_and_caught(self):
        with pytest.raises(OptimizerError, match="test"):
            raise OptimizerError("test")


# ===================================================================
# Server error hierarchy
# ===================================================================


@pytest.mark.unit
class TestServerErrors:
    def test_server_error_inherits_optimizer_error(self):
        assert issubclass(ServerError, OptimizerError)


# ===================================================================
# Other error classes
# ===================================================================


@pytest.mark.unit
class TestBenchOOMError:
    def test_inherits_optimizer_error(self):
        assert issubclass(BenchOOMError, OptimizerError)

    def test_construction(self):
        err = BenchOOMError("llama-bench OOM")
        assert "OOM" in str(err)


@pytest.mark.unit
class TestBaselineFailure:
    def test_inherits_optimizer_error(self):
        assert issubclass(BaselineFailure, OptimizerError)

    def test_construction_and_str(self):
        err = BaselineFailure("Baseline server failed in Core Engine.")
        assert "Core Engine" in str(err)

    def test_can_be_raised_and_caught_as_optimizer_error(self):
        with pytest.raises(OptimizerError):
            raise BaselineFailure("baseline down")

    def test_repr_contains_class_name(self):
        err = BaselineFailure("test")
        assert "BaselineFailure" in repr(err)
