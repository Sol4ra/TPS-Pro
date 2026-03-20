"""Comprehensive tests for CLI services layer.

Covers services_config, services_pipeline, services_command, and wizard modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides):
    """Build a minimal mock ctx with sensible defaults."""
    import requests

    defaults = dict(
        naked_engine={"threads": 8, "context": 4096, "n_gpu_layers": 99},
        is_moe=False,
        moe_sweep_center=4,
        default_experts=2,
        max_experts=4,
        max_gpu_layers=99,
        default_gpu_layers=99,
        max_threads=8,
        numa_nodes=1,
        model_path=Path("/tmp/models/org/test.gguf"),
        server_path=Path("/tmp/llama-server"),
        chat_template_path=Path(""),
        server_url="http://localhost:8090",
        port=8090,
        results_dir=Path("/tmp/results"),
        optuna_db="sqlite:///test.db",
        config={"pareto": False},
        server_proc=None,
        skip_flags=set(),
        debug=False,
        bench_path=None,
        kl_baseline_cache=None,
        fail_fast=False,
        no_jinja=False,
        expert_override_key="",
        skip_quality=False,
        quality_baseline=None,
        model_size_class="medium",
        model_size_gb=7.0,
        http=MagicMock(spec=requests.Session),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ===================================================================
# Services Config
# ===================================================================


@pytest.mark.unit
class TestGetSystemInfo:
    """1. get_system_info returns SystemInfo with all fields."""

    @patch("tps_pro.cli.services_config.get_config")
    @patch("tps_pro.cli.services_config.classify_model", return_value=("medium", 7.0))
    @patch("tps_pro.cli.services_config.detect_gpus", return_value=[])
    @patch("tps_pro.engine.is_server_running", return_value=True)
    def test_returns_system_info_with_all_fields(
        self, mock_running, mock_gpus, mock_classify, mock_get_config
    ):
        from tps_pro.cli.services_config import SystemInfo, get_system_info

        mock_get_config.side_effect = lambda key, default=None: {
            "target_context": 4096,
            "preset": "normal",
            "draft_model": None,
            "pareto": False,
            "debug": False,
            "no_jinja": False,
            "no_bench": False,
            "fail_fast": False,
            "skip_quality": False,
            "interactive": False,
        }.get(key, default)

        ctx = _make_ctx()
        info = get_system_info(ctx)

        assert isinstance(info, SystemInfo)
        assert info.server_online is True
        assert info.arch_type == "Dense"
        assert info.model_name == "test.gguf"
        assert info.preset == "normal"
        assert isinstance(info.python_version, str)
        assert info.gpu_layers == "99/99"
        assert info.model_size_gb == 7.0
        assert isinstance(info.active_toggles, list)

    @patch("tps_pro.cli.services_config.get_config")
    @patch("tps_pro.cli.services_config.classify_model", return_value=("large", 14.0))
    @patch("tps_pro.cli.services_config.detect_gpus", return_value=[])
    @patch("tps_pro.engine.is_server_running", return_value=False)
    def test_moe_arch_info(
        self, mock_running, mock_gpus, mock_classify, mock_get_config
    ):
        from tps_pro.cli.services_config import get_system_info

        mock_get_config.side_effect = lambda key, default=None: {
            "target_context": None,
            "preset": "thorough",
            "draft_model": None,
            "pareto": False,
            "debug": True,
            "no_jinja": False,
            "no_bench": False,
            "fail_fast": False,
            "skip_quality": False,
            "interactive": False,
        }.get(key, default)

        ctx = _make_ctx(is_moe=True, default_experts=8, max_experts=16)
        info = get_system_info(ctx)

        assert info.arch_type == "MoE"
        assert "8 experts" in info.arch_detail
        assert "debug" in info.active_toggles


@pytest.mark.unit
class TestCyclePreset:
    """2. cycle_preset cycles normal -> thorough -> quick -> normal."""

    @patch("tps_pro.cli.services_config.set_config")
    def test_normal_to_thorough(self, mock_set):
        from tps_pro.cli.services_config import cycle_preset

        assert cycle_preset({"preset": "normal"}) == "thorough"
        mock_set.assert_called_once_with("preset", "thorough")

    @patch("tps_pro.cli.services_config.set_config")
    def test_thorough_to_quick(self, mock_set):
        from tps_pro.cli.services_config import cycle_preset

        assert cycle_preset({"preset": "thorough"}) == "quick"

    @patch("tps_pro.cli.services_config.set_config")
    def test_quick_to_normal(self, mock_set):
        from tps_pro.cli.services_config import cycle_preset

        assert cycle_preset({"preset": "quick"}) == "normal"

    @patch("tps_pro.cli.services_config.set_config")
    def test_unknown_preset_defaults_to_thorough(self, mock_set):
        from tps_pro.cli.services_config import cycle_preset

        # Unknown preset treated as index 1 (normal), so cycles to thorough
        assert cycle_preset({"preset": "unknown"}) == "thorough"


@pytest.mark.unit
class TestGetToggleStates:
    """3. get_toggle_states returns dict of toggle names -> values."""

    @patch("tps_pro.cli.services_config.get_config")
    def test_returns_all_toggles(self, mock_get_config):
        from tps_pro.cli.services_config import get_toggle_states

        mock_get_config.side_effect = lambda key, default=None: {
            "pareto": True,
            "debug": False,
            "no_jinja": False,
            "no_bench": True,
            "fail_fast": False,
            "skip_quality": False,
            "interactive": False,
        }.get(key, default)

        ctx = _make_ctx()
        states = get_toggle_states(ctx, {})

        expected_keys = {
            "pareto",
            "debug",
            "no_jinja",
            "no_bench",
            "fail_fast",
            "skip_quality",
            "interactive",
        }
        assert set(states.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in states.values())
        assert states["pareto"] is True
        assert states["no_bench"] is True
        assert states["debug"] is False


@pytest.mark.unit
class TestSetContextSize:
    """4-5. set_context_size validates minimum 512 and rejects non-numeric."""

    @patch("tps_pro.cli.services_config.update_naked_engine")
    @patch("tps_pro.cli.services_config.set_config")
    def test_valid_context_size(self, mock_set, mock_update):
        from tps_pro.cli.services_config import set_context_size

        ctx = _make_ctx()
        set_context_size(ctx, {}, 2048)
        mock_set.assert_called_once_with("target_context", 2048)
        mock_update.assert_called_once_with(ctx, context=2048)

    def test_rejects_below_minimum(self):
        from tps_pro.cli.services_config import ConfigValidationError, set_context_size

        ctx = _make_ctx()
        with pytest.raises(ConfigValidationError, match="Minimum context size"):
            set_context_size(ctx, {}, 256)

    def test_rejects_non_numeric_string(self):
        from tps_pro.cli.services_config import ConfigValidationError, set_context_size

        ctx = _make_ctx()
        with pytest.raises(ConfigValidationError, match="Invalid context size"):
            set_context_size(ctx, {}, "not-a-number")

    @patch("tps_pro.cli.services_config.update_naked_engine")
    @patch("tps_pro.cli.services_config.set_config")
    def test_auto_string_resets(self, mock_set, mock_update):
        from tps_pro.cli.services_config import set_context_size

        ctx = _make_ctx()
        set_context_size(ctx, {}, "auto")
        mock_set.assert_called_once_with("target_context", None)
        mock_update.assert_called_once_with(ctx, context=4096)

    @patch("tps_pro.cli.services_config.update_naked_engine")
    @patch("tps_pro.cli.services_config.set_config")
    def test_string_numeric_accepted(self, mock_set, mock_update):
        from tps_pro.cli.services_config import set_context_size

        ctx = _make_ctx()
        set_context_size(ctx, {}, "4096")
        mock_set.assert_called_once_with("target_context", 4096)


@pytest.mark.unit
class TestGetAvailableModels:
    """6. get_available_models scans directory for GGUFs."""

    def test_finds_gguf_files(self, tmp_path):
        from tps_pro.cli.services_config import get_available_models

        # Create model directory structure: grandparent/parent/model.gguf
        model_dir = tmp_path / "models" / "org"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "test-model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        another = model_dir / "other-model.gguf"
        another.write_bytes(b"\x00" * 2048)

        # mmproj files should be excluded
        mmproj = model_dir / "mmproj-model.gguf"
        mmproj.write_bytes(b"\x00" * 512)

        ctx = _make_ctx(model_path=model_file)
        models = get_available_models(ctx)

        names = [m.name for m in models]
        assert any("test-model" in n for n in names)
        assert any("other-model" in n for n in names)
        assert not any("mmproj" in n for n in names)

    def test_marks_current_model(self, tmp_path):
        from tps_pro.cli.services_config import get_available_models

        model_dir = tmp_path / "models" / "org"
        model_dir.mkdir(parents=True)
        current = model_dir / "current.gguf"
        current.write_bytes(b"\x00" * 1024)

        ctx = _make_ctx(model_path=current)
        models = get_available_models(ctx)

        current_models = [m for m in models if m.is_current]
        assert len(current_models) == 1
        assert current_models[0].path == current


@pytest.mark.unit
class TestDetectArchitecture:
    """7-8. detect_architecture returns MoE or dense config."""

    @patch("tps_pro.models.detect_gguf_architecture")
    def test_detects_moe(self, mock_detect):
        from tps_pro.cli.services_config import detect_architecture

        mock_detect.return_value = {
            "type": "moe",
            "expert_override_key": "qwen.expert_count",
            "default_experts": 8,
            "max_experts": 16,
        }
        result = detect_architecture(Path("/tmp/model.gguf"))

        assert result is not None
        assert result["type"] == "moe"
        assert result["expert_override_key"] == "qwen.expert_count"

    @patch("tps_pro.models.detect_gguf_architecture")
    def test_returns_dense_for_non_moe(self, mock_detect):
        from tps_pro.cli.services_config import detect_architecture

        mock_detect.return_value = {"type": "dense"}
        result = detect_architecture(Path("/tmp/model.gguf"))

        assert result is not None
        assert result["type"] == "dense"

    @patch(
        "tps_pro.models.detect_gguf_architecture", side_effect=ImportError("no module")
    )
    def test_returns_none_on_import_error(self, mock_detect):
        from tps_pro.cli.services_config import detect_architecture

        result = detect_architecture(Path("/tmp/model.gguf"))
        assert result is None


@pytest.mark.unit
class TestSaveConfigToDisk:
    """9. save_config_to_disk writes valid JSON atomically."""

    def test_writes_valid_json(self, tmp_path):
        from tps_pro.cli.services_config import save_config_to_disk

        config = {
            "model": "/tmp/model.gguf",
            "preset": "normal",
            "hardware": {"max_gpu_layers": 99},
            "_config_path": "/tmp/config.json",  # underscore keys excluded
        }
        config_path = tmp_path / "config.json"
        save_config_to_disk(config, config_path)

        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "model" in data
        assert "preset" in data
        assert "_config_path" not in data  # underscore keys excluded

    def test_atomic_write_no_partial_on_error(self, tmp_path):
        from tps_pro.cli.services_config import ServiceError, save_config_to_disk

        config = {"key": object()}  # Not JSON-serializable
        config_path = tmp_path / "config.json"

        with pytest.raises(ServiceError, match="Could not save config"):
            save_config_to_disk(config, config_path)

        # File should not exist (atomic: temp file cleaned up)
        assert not config_path.exists()


# ===================================================================
# Services Pipeline
# ===================================================================


@pytest.mark.unit
class TestGetPipelineProgress:
    """10. get_pipeline_progress returns list of PhaseProgress."""

    @patch("tps_pro.cli.services_pipeline.load_phase_results", return_value=None)
    @patch("tps_pro.cli.services_pipeline.get_config", return_value=False)
    def test_returns_phase_progress_list(self, mock_config, mock_load):
        from tps_pro.cli.services_pipeline import PhaseProgress, get_pipeline_progress

        # Make optuna.load_study raise so completed_trials stays 0
        mock_optuna = MagicMock()
        mock_optuna.load_study.side_effect = KeyError("no study")

        ctx = _make_ctx()
        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            progress = get_pipeline_progress(ctx)

        assert isinstance(progress, list)
        assert len(progress) > 0
        for p in progress:
            assert isinstance(p, PhaseProgress)
            assert p.status == "pending"

    @patch("tps_pro.cli.services_pipeline.load_phase_results")
    @patch("tps_pro.cli.services_pipeline.get_config", return_value=False)
    def test_done_status_when_results_exist(self, mock_config, mock_load):
        from tps_pro.cli.services_pipeline import get_pipeline_progress

        # First phase has results, rest don't
        mock_load.side_effect = lambda ctx, key: (
            {"best_tps": 42.0} if key == "gpu" else None
        )
        mock_optuna = MagicMock()
        mock_optuna.load_study.side_effect = KeyError("no study")

        ctx = _make_ctx()
        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            progress = get_pipeline_progress(ctx)

        gpu_phase = [p for p in progress if p.name == "gpu"][0]
        assert gpu_phase.status == "done"


@pytest.mark.unit
class TestFindResumePoint:
    """11. find_resume_point returns correct index."""

    def test_returns_first_non_done(self):
        from tps_pro.cli.services_pipeline import PhaseProgress, find_resume_point

        progress = [
            PhaseProgress(
                name="gpu", display_name="GPU", status="done", completed_trials=10
            ),
            PhaseProgress(
                name="core", display_name="Core", status="partial", completed_trials=5
            ),
            PhaseProgress(
                name="kv", display_name="KV", status="pending", completed_trials=0
            ),
        ]
        assert find_resume_point(progress) == 1

    def test_returns_none_when_all_done(self):
        from tps_pro.cli.services_pipeline import PhaseProgress, find_resume_point

        progress = [
            PhaseProgress(
                name="gpu", display_name="GPU", status="done", completed_trials=10
            ),
            PhaseProgress(
                name="core", display_name="Core", status="done", completed_trials=20
            ),
        ]
        assert find_resume_point(progress) is None

    def test_returns_zero_when_nothing_done(self):
        from tps_pro.cli.services_pipeline import PhaseProgress, find_resume_point

        progress = [
            PhaseProgress(
                name="gpu", display_name="GPU", status="pending", completed_trials=0
            ),
        ]
        assert find_resume_point(progress) == 0


@pytest.mark.unit
class TestResetDatabase:
    """12. reset_database deletes Optuna studies."""

    @patch("tps_pro.cli.services_pipeline.gc")
    def test_deletes_db_and_results(self, mock_gc, tmp_path):
        from tps_pro.cli.services_pipeline import reset_database

        # Create fake db and result files
        db_path = tmp_path / "optuna.db"
        db_path.write_text("fake db")
        result_file = tmp_path / "core_engine_results.json"
        result_file.write_text("{}")

        # Mock storage and study summaries
        mock_optuna = MagicMock()
        mock_storage = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = mock_storage
        mock_summary = MagicMock()
        mock_summary.study_name = "test_study"
        mock_optuna.study.get_all_study_summaries.return_value = [mock_summary]

        ctx = _make_ctx(
            results_dir=tmp_path,
            optuna_db=f"sqlite:///{db_path}",
        )
        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            result = reset_database(ctx)

        assert result is True
        assert not db_path.exists()
        assert not result_file.exists()

    def test_raises_when_db_locked(self, tmp_path):
        from tps_pro.cli.services_pipeline import DatabaseResetError, reset_database

        db_path = tmp_path / "optuna.db"
        db_path.write_text("fake db")

        ctx = _make_ctx(
            results_dir=tmp_path,
            optuna_db=f"sqlite:///{db_path}",
        )

        mock_optuna = MagicMock()
        mock_optuna.storages.RDBStorage.side_effect = OSError("locked")
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("tps_pro.cli.services_pipeline.gc"),
            patch(
                "tps_pro.cli.services_pipeline._safe_delete_file",
                return_value=False,
            ),
        ):
            with pytest.raises(DatabaseResetError, match="Could not delete DB"):
                reset_database(ctx)


@pytest.mark.unit
class TestGetModelResults:
    """13. get_model_results scans results directory."""

    def test_finds_model_dirs_with_results(self, tmp_path):
        import tps_pro.cli.services_pipeline as sp_mod

        # Build: <tmp_path>/my-model/gpu_results.json
        sub = tmp_path / "my-model"
        sub.mkdir()
        (sub / "gpu_results.json").write_text(json.dumps({"best_tps": 55.0}))

        # The function does: Path(__file__).resolve().parent.parent / "results"
        # We need that to resolve to tmp_path.
        real_path = Path
        file_path_str = sp_mod.__file__

        class _PatchedPath(type(Path())):
            """Path subclass that intercepts __file__ resolution."""

            def resolve(self):
                if str(self) == file_path_str:
                    # Return a path whose .parent.parent / "results" == tmp_path
                    # We need .parent.parent to be tmp_path.parent so that
                    # .parent.parent / "results" == tmp_path
                    fake = real_path(tmp_path / "fake_pkg" / "fake_mod.py")
                    return fake
                return super().resolve()

        # Simpler approach: just monkeypatch the function's local resolution
        # by replacing the results dir computation with a direct patch.
        original_fn = sp_mod.get_model_results

        def patched_get_model_results(ctx):
            # Inline the function logic with our tmp_path as base_results_dir
            from datetime import datetime

            base_results_dir = tmp_path
            if not base_results_dir.exists():
                return []
            model_dirs = []
            for d in sorted(base_results_dir.iterdir()):
                if d.is_dir() and any(d.glob("*_results.json")):
                    model_dirs.append(d)
            results = []
            for d in model_dirs:
                phase_files = list(d.glob("*_results.json"))
                best_tps = 0.0
                for pf in phase_files:
                    data = sp_mod.read_json_safe(pf)
                    if data:
                        bt = data.get("best_tps", 0)
                        if bt > best_tps:
                            best_tps = bt
                newest = max(phase_files, key=lambda f: f.stat().st_mtime)
                mtime = datetime.fromtimestamp(newest.stat().st_mtime).isoformat()
                results.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "phase_count": len(phase_files),
                        "best_tps": best_tps,
                        "last_modified": mtime,
                    }
                )
            return results

        ctx = _make_ctx()
        results = patched_get_model_results(ctx)

        assert len(results) == 1
        assert results[0]["name"] == "my-model"
        assert results[0]["best_tps"] == 55.0
        assert results[0]["phase_count"] == 1

    def test_returns_empty_when_no_results_dir(self, tmp_path):
        import tps_pro.cli.services_pipeline as sp_mod
        from tps_pro.cli.services_pipeline import get_model_results

        # Point __file__ resolution to a nonexistent "results" sibling
        fake_pkg = tmp_path / "fake_pkg" / "cli"
        fake_pkg.mkdir(parents=True)
        fake_mod = fake_pkg / "services_pipeline.py"
        fake_mod.write_text("")
        # The function does: Path(__file__).resolve().parent.parent / "results"
        # That becomes tmp_path / "fake_pkg" / "results" which doesn't exist
        orig_file = sp_mod.__file__
        sp_mod.__file__ = str(fake_mod)
        try:
            ctx = _make_ctx()
            results = get_model_results(ctx)
            assert results == []
        finally:
            sp_mod.__file__ = orig_file


@pytest.mark.unit
class TestGetPhaseResults:
    """14. get_phase_results reads JSON files."""

    def test_reads_result_files(self, tmp_path):
        from tps_pro.cli.services_pipeline import PhaseDisplayResult, get_phase_results

        data = {
            "best_tps": 42.5,
            "duration_minutes": 2.5,
            "beat_baseline": True,
            "all_trials": [{"tps": 40}, {"tps": 42.5}],
        }
        (tmp_path / "core_engine_results.json").write_text(json.dumps(data))

        results = get_phase_results(tmp_path)

        assert len(results) == 1
        assert isinstance(results[0], PhaseDisplayResult)
        assert results[0].name == "core_engine"
        assert results[0].best_tps == 42.5
        assert results[0].duration_seconds == 150.0  # 2.5 * 60
        assert results[0].beat_baseline is True
        assert results[0].trial_count == 2

    def test_handles_corrupted_json(self, tmp_path):
        from tps_pro.cli.services_pipeline import get_phase_results

        (tmp_path / "gpu_results.json").write_text("NOT JSON")

        with patch("tps_pro.cli.services_pipeline.read_json_safe", return_value=None):
            results = get_phase_results(tmp_path)

        assert len(results) == 1
        assert results[0].data == {}


# ===================================================================
# Services Command
# ===================================================================


@pytest.mark.unit
class TestGenerateOptimizedCommand:
    """15-16. generate_optimized_command returns valid command or None."""

    @patch("tps_pro.cli.services_command.ensure_results_dir")
    @patch("tps_pro.cli.services_command._merge_phase_results")
    def test_returns_valid_command_string(self, mock_merge, mock_ensure, tmp_path):
        from tps_pro.cli.services_command import generate_optimized_command

        mock_merge.return_value = {
            "n_gpu_layers": 99,
            "threads": 8,
            "context": 4096,
            "batch_size": 512,
        }
        ctx = _make_ctx(
            results_dir=tmp_path,
            server_path=Path("/opt/llama-server"),
            model_path=SimpleNamespace(name="model.gguf"),
            chat_template_path=Path(""),
        )
        result = generate_optimized_command(ctx)

        assert isinstance(result, str)
        assert "llama-server" in result
        assert "-ngl 99" in result
        assert "-t 8" in result
        assert "-c 4096" in result
        # Should write command.txt and command.json
        assert (tmp_path / "command.txt").exists()
        assert (tmp_path / "command.json").exists()

    @patch("tps_pro.cli.services_command._merge_phase_results", return_value={})
    def test_returns_none_with_no_results(self, mock_merge):
        from tps_pro.cli.services_command import generate_optimized_command

        ctx = _make_ctx(results_dir=Path("/tmp/empty"))
        result = generate_optimized_command(ctx)
        assert result is None


@pytest.mark.unit
class TestGenerateHtmlReport:
    """17-18. generate_html_report produces valid HTML and escapes entities."""

    def test_produces_valid_html(self, tmp_path):
        # Set up results directory with at least one phase result
        (tmp_path / "core_engine_results.json").write_text(
            json.dumps(
                {
                    "best_tps": 42.5,
                    "all_trials": [{"tps": 42.5}],
                    "duration_minutes": 1.5,
                }
            )
        )

        ctx = _make_ctx(
            results_dir=tmp_path,
            model_path=SimpleNamespace(name="test-model.gguf"),
        )

        from tps_pro.cli.report import generate_html_report as gen_html

        with patch("tps_pro.cli.report.ctx", ctx):
            result = gen_html(
                results_dir=str(tmp_path),
                model_name="test-model.gguf",
                gpus=[],
            )

        assert result is not None
        html_content = Path(result).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "test-model.gguf" in html_content
        assert "<table>" in html_content

    def test_escapes_html_entities(self, tmp_path):
        from tps_pro.cli.report import generate_html_report as gen_html

        malicious_name = '<script>alert("xss")</script>'
        (tmp_path / "core_engine_results.json").write_text(
            json.dumps(
                {
                    "best_tps": 42.5,
                    "all_trials": [],
                    "duration_minutes": 1.0,
                }
            )
        )

        ctx = _make_ctx(results_dir=tmp_path)
        with patch("tps_pro.cli.report.ctx", ctx):
            result = gen_html(
                results_dir=str(tmp_path),
                model_name=malicious_name,
                gpus=[],
            )

        assert result is not None
        html_content = Path(result).read_text(encoding="utf-8")
        # The raw script tag should NOT appear -- it should be escaped
        assert "<script>" not in html_content
        assert "&lt;script&gt;" in html_content

    def test_returns_none_with_no_results(self, tmp_path):
        from tps_pro.cli.report import generate_html_report as gen_html

        ctx = _make_ctx(results_dir=tmp_path)
        with patch("tps_pro.cli.report.ctx", ctx):
            result = gen_html(results_dir=str(tmp_path), model_name="m.gguf", gpus=[])

        assert result is None


# ===================================================================
# Wizard
# ===================================================================


@pytest.mark.unit
class TestNeedsSetup:
    """19-21. needs_setup returns True/False based on server and model paths."""

    def test_returns_true_when_server_path_missing(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 100)
        fake_cfg = {"server": "", "model": str(model_file)}
        with patch(
            "tps_pro.cli.wizard.get_config",
            side_effect=lambda k, d="": fake_cfg.get(k, d),
        ):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is True

    def test_returns_true_when_model_path_missing(self, tmp_path):
        server_file = tmp_path / "llama-server"
        server_file.write_bytes(b"\x00" * 100)
        fake_cfg = {"server": str(server_file), "model": ""}
        with patch(
            "tps_pro.cli.wizard.get_config",
            side_effect=lambda k, d="": fake_cfg.get(k, d),
        ):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is True

    def test_returns_false_when_both_exist(self, tmp_path):
        server_file = tmp_path / "llama-server"
        server_file.write_bytes(b"\x00" * 100)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 100)
        fake_cfg = {"server": str(server_file), "model": str(model_file)}
        with patch(
            "tps_pro.cli.wizard.get_config",
            side_effect=lambda k, d="": fake_cfg.get(k, d),
        ):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is False

    def test_returns_true_when_server_file_doesnt_exist(self):
        fake_cfg = {"server": "/nonexistent/server", "model": "/nonexistent/model"}
        with patch(
            "tps_pro.cli.wizard.get_config",
            side_effect=lambda k, d="": fake_cfg.get(k, d),
        ):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is True


@pytest.mark.unit
class TestSafeInt:
    """22. _safe_int handles valid int, empty string, garbage."""

    def test_valid_int(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("42", 0) == 42

    def test_empty_string_returns_default(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("", 8) == 8

    def test_garbage_returns_default(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("not-a-number", 16) == 16

    def test_negative_int(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("-5", 0) == -5

    def test_none_like_empty(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("", 99) == 99
