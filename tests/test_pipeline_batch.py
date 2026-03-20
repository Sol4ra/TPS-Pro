"""Tests for pipeline.py: batch_optimize() orchestration.

All phase functions and external dependencies are mocked to test
orchestration logic in isolation — no servers, no models, no Optuna.
"""

import time
from unittest.mock import patch

import pytest

from tps_pro.pipeline import (
    batch_optimize,
)

# ---------------------------------------------------------------------------
# Module paths for patching (pipeline.py re-imports these names)
# ---------------------------------------------------------------------------
_P = "tps_pro.pipeline"


# ===================================================================
# batch_optimize tests
# ===================================================================


class TestBatchOptimize:
    """Tests for batch_optimize orchestration."""

    def _create_gguf_files(self, tmp_path, names):
        """Create dummy .gguf files in tmp_path."""
        paths = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"\x00" * 100)
            paths.append(p)
        return paths

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_iterates_over_multiple_models(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """batch_optimize processes each GGUF file."""
        self._create_gguf_files(
            tmp_path, ["model_a.gguf", "model_b.gguf", "model_c.gguf"]
        )

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path))

        assert mock_pipeline.call_count == 3

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_skips_non_gguf_files(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """Non-GGUF files in the directory are ignored."""
        self._create_gguf_files(tmp_path, ["model.gguf"])
        (tmp_path / "readme.txt").write_text("not a model")
        (tmp_path / "config.json").write_text("{}")

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path))

        assert mock_pipeline.call_count == 1

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_individual_model_failure_continues(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """If one model fails, batch continues to the next."""
        self._create_gguf_files(tmp_path, ["model_a.gguf", "model_b.gguf"])

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        # First model fails, second succeeds
        mock_pipeline.side_effect = [RuntimeError("GPU exploded"), None]

        batch_optimize(str(tmp_path))

        assert mock_pipeline.call_count == 2

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_creates_per_model_results_dir(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """Each model gets its own results directory."""
        self._create_gguf_files(tmp_path, ["my-model.gguf"])

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path))

        expected_dir = tmp_path / "optimize-results-my-model"
        assert expected_dir.is_dir()

    @pytest.mark.unit
    def test_empty_directory_logs_error(self, tmp_path):
        """batch_optimize on an empty directory does not crash."""
        # Should not raise — just logs an error
        batch_optimize(str(tmp_path))

    @pytest.mark.unit
    def test_invalid_directory_logs_error(self):
        """batch_optimize on a nonexistent directory does not crash."""
        batch_optimize("/nonexistent/path/does/not/exist")

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_skip_existing_with_results(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """skip_existing=True skips models that already have results."""
        self._create_gguf_files(tmp_path, ["model_a.gguf"])

        # Pre-create a results directory with a results file
        results_dir = tmp_path / "optimize-results-model_a"
        results_dir.mkdir()
        (results_dir / "gpu_results.json").write_text('{"best_ngl": 99}')

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path), skip_existing=True)

        mock_pipeline.assert_not_called()

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_filters_mmproj_and_embedding_models(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """mmproj files and models in embedding/reranker dirs are skipped."""
        # Regular model
        self._create_gguf_files(tmp_path, ["real-model.gguf"])
        # mmproj model (should be filtered)
        (tmp_path / "model-mmproj-fp16.gguf").write_bytes(b"\x00" * 100)
        # Embedding directory model (should be filtered)
        embed_dir = tmp_path / "embedding"
        embed_dir.mkdir()
        (embed_dir / "embed-model.gguf").write_bytes(b"\x00" * 100)
        # Reranker directory model (should be filtered)
        reranker_dir = tmp_path / "reranker"
        reranker_dir.mkdir()
        (reranker_dir / "rerank-model.gguf").write_bytes(b"\x00" * 100)

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path))

        # Only the real model should be processed
        assert mock_pipeline.call_count == 1

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_baseline_failure_skips_model(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """BaselineFailure from run_full_pipeline is caught and model is skipped."""
        from tps_pro.errors import BaselineFailure

        self._create_gguf_files(tmp_path, ["model_a.gguf", "model_b.gguf"])

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        mock_pipeline.side_effect = [BaselineFailure("Server won't start"), None]

        batch_optimize(str(tmp_path))

        # Both models attempted — first failed with BaselineFailure, second succeeded
        assert mock_pipeline.call_count == 2

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_kill_server_called_in_finally(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """kill_server is called after each model (finally block)."""
        self._create_gguf_files(tmp_path, ["model.gguf"])

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path))

        mock_kill.assert_called()

    @pytest.mark.unit
    @patch(f"{_P}.run_full_pipeline")
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.ensure_results_dir")
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.detect_model_layers", return_value=32)
    @patch(f"{_P}.detect_gguf_architecture", return_value={"type": "dense"})
    @patch(f"{_P}.classify_model", return_value=("medium", 4.0))
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.ctx")
    def test_timeout_passed_to_pipeline(
        self,
        mock_ctx,
        mock_gpus,
        mock_classify,
        mock_arch,
        mock_layers,
        mock_skip_flags,
        mock_load,
        mock_ensure,
        mock_kill,
        mock_pipeline,
        tmp_path,
    ):
        """Per-model timeout is converted to a deadline and passed to
        run_full_pipeline."""
        self._create_gguf_files(tmp_path, ["model.gguf"])

        mock_ctx.config = {}
        mock_ctx.quality_baseline = None
        mock_ctx.kl_baseline_cache = None
        mock_ctx.no_jinja = False
        mock_ctx.default_gpu_layers = 99
        mock_ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        mock_ctx.skip_flags = set()
        mock_ctx.results_dir = tmp_path / "results"
        mock_ctx.is_moe = False

        batch_optimize(str(tmp_path), timeout_minutes=30)

        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert "deadline" in call_kwargs
        # Deadline should be roughly now + 30 minutes
        assert call_kwargs["deadline"] > time.time() - 10
        assert call_kwargs["deadline"] < time.time() + 30 * 60 + 60
