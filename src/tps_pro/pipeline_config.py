"""Configurable pipeline system.

Allows users to:
1. Set global flags that apply to every server boot
   and get stripped from phase search pools
2. Reorder phases in any order
3. Enable/disable phases
4. Control what each phase searches (add/remove params from test pools)
5. Lock specific params to fixed values per phase

Config is stored in results/<model>/pipeline-config.json and can be edited
via the TUI or by hand.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default phase definitions with their configurable options
DEFAULT_PHASES = [
    {
        "phase": "gpu_offload",
        "display_name": "GPU Offload",
        "enabled": True,
        "description": "Find optimal GPU layer count",
    },
    {
        "phase": "moe_sweep",
        "display_name": "MoE Threads",
        "enabled": True,
        "moe_only": True,
        "range": [8, 32],
        "step": 2,
        "description": "Find optimal MoE CPU thread count",
    },
    {
        "phase": "kv_context_sweep",
        "display_name": "KV + Context Sweep",
        "enabled": True,
        "kv_types": ["f16", "q8_0", "q4_0"],
        "description": "Find optimal KV cache type and max context",
    },
    {
        "phase": "ab_toggles",
        "display_name": "A/B Toggles",
        "enabled": True,
        "test_flags": [
            "op_offload", "prio", "prio_batch",
            "no_mmap", "mlock", "repack", "swa_full",
            "numa", "cpu_strict", "cpu_strict_batch",
        ],
        "description": "Independent binary flag sweeps",
    },
    {
        "phase": "core_engine",
        "display_name": "Core Engine",
        "enabled": True,
        "trials": 100,
        "search_params": [
            "threads", "threads_batch", "batch_size",
            "ubatch_size", "flash_attn", "poll", "poll_batch",
        ],
        "lock": {},
        "description": "Focused TPE search for correlated params",
    },
    {
        "phase": "speculation",
        "display_name": "Speculation",
        "enabled": True,
        "trials": 40,
        "search_params": [
            "spec_type", "spec_ngram_n", "spec_ngram_m",
            "spec_ngram_min_hits", "draft_max", "draft_min",
            "draft_p_min",
        ],
        "lock": {},
        "description": "Speculative decoding params",
    },
    {
        "phase": "workload_sim",
        "display_name": "Workload Sim",
        "enabled": True,
        "description": "Hot-cache and concurrent load testing",
    },
    {
        "phase": "quality",
        "display_name": "Quality/Sampling",
        "enabled": True,
        "trials": 60,
        "description": "Sampling parameter optimization",
    },
]


@dataclass
class PhaseConfig:
    """Configuration for a single pipeline phase."""

    phase: str
    display_name: str
    enabled: bool = True
    moe_only: bool = False
    description: str = ""
    trials: int | None = None
    search_params: list[str] = field(default_factory=list)
    test_flags: list[str] = field(default_factory=list)
    lock: dict[str, Any] = field(default_factory=dict)
    kv_types: list[str] = field(default_factory=list)
    range: list[int] = field(default_factory=list)
    step: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhaseConfig:
        """Build from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:  # noqa: C901
        """Serialize to dict, omitting empty/default fields."""
        result = {"phase": self.phase, "display_name": self.display_name}
        if not self.enabled:
            result["enabled"] = False
        if self.moe_only:
            result["moe_only"] = True
        if self.description:
            result["description"] = self.description
        if self.trials is not None:
            result["trials"] = self.trials
        if self.search_params:
            result["search_params"] = self.search_params
        if self.test_flags:
            result["test_flags"] = self.test_flags
        if self.lock:
            result["lock"] = self.lock
        if self.kv_types:
            result["kv_types"] = self.kv_types
        if self.range:
            result["range"] = self.range
        if self.step != 2:  # noqa: PLR2004
            result["step"] = self.step
        return result


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the KV sweep scoring formula.

    score = (tps * tps) + (context_bonus * context) + (pp_speed * pp_speed)
    """

    tps: float = 0.5
    context: float = 0.3
    pp_speed: float = 0.2

    def to_dict(self) -> dict[str, float]:
        return {"tps": self.tps, "context": self.context, "pp_speed": self.pp_speed}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoringWeights:
        return cls(
            tps=float(d.get("tps", 0.5)),
            context=float(d.get("context", 0.3)),
            pp_speed=float(d.get("pp_speed", 0.2)),
        )


_DEFAULT_PRESETS: dict[str, float] = {
    "quick": 0.5,
    "normal": 1.0,
    "thorough": 1.5,
}


@dataclass
class PipelineConfig:
    """Full pipeline configuration with global flags and phase ordering."""

    global_flags: dict[str, Any] = field(default_factory=dict)
    phases: list[PhaseConfig] = field(default_factory=list)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    presets: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PRESETS)
    )

    @classmethod
    def default(cls, is_moe: bool = False) -> PipelineConfig:
        """Create default config, filtering MoE phases for dense models."""
        phases = []
        for p in DEFAULT_PHASES:
            pc = PhaseConfig.from_dict(p)
            if pc.moe_only and not is_moe:
                pc.enabled = False
            phases.append(pc)
        return cls(
            global_flags={},
            phases=phases,
            scoring_weights=ScoringWeights(),
            presets=dict(_DEFAULT_PRESETS),
        )

    @classmethod
    def load(cls, config_path: Path, is_moe: bool = False) -> PipelineConfig:
        """Load config from JSON file, falling back to defaults."""
        if config_path.exists():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                return cls._from_dict(raw, is_moe)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Invalid pipeline config (%s) — using defaults", e)
        return cls.default(is_moe)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any], is_moe: bool) -> PipelineConfig:
        """Parse from a raw dict."""
        global_flags = raw.get("global_flags", {})

        # Build phase configs from the saved order
        phases = []
        for p in raw.get("pipeline", []):
            pc = PhaseConfig.from_dict(p)
            if pc.moe_only and not is_moe:
                pc.enabled = False
            phases.append(pc)

        # If saved config has fewer phases than defaults, append missing ones
        saved_names = {p.phase for p in phases}
        for p in DEFAULT_PHASES:
            if p["phase"] not in saved_names:
                pc = PhaseConfig.from_dict(p)
                if pc.moe_only and not is_moe:
                    pc.enabled = False
                phases.append(pc)

        # Parse scoring weights if present
        raw_weights = raw.get("scoring_weights")
        scoring_weights = (
            ScoringWeights.from_dict(raw_weights)
            if isinstance(raw_weights, dict)
            else ScoringWeights()
        )

        # Parse presets if present, otherwise use defaults
        raw_presets = raw.get("presets")
        presets = (
            {k: float(v) for k, v in raw_presets.items()}
            if isinstance(raw_presets, dict)
            else dict(_DEFAULT_PRESETS)
        )

        return cls(
            global_flags=global_flags,
            phases=phases,
            scoring_weights=scoring_weights,
            presets=presets,
        )

    def save(self, config_path: Path) -> None:
        """Save config to JSON file."""
        data = self.to_dict()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(config_path)
        logger.info("Pipeline config saved to %s", config_path)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full config."""
        result: dict[str, Any] = {
            "global_flags": self.global_flags,
            "pipeline": [p.to_dict() for p in self.phases],
        }
        # Only include scoring_weights if non-default
        default_weights = ScoringWeights()
        if self.scoring_weights != default_weights:
            result["scoring_weights"] = (
                self.scoring_weights.to_dict()
            )
        # Only include presets if non-default
        if self.presets != _DEFAULT_PRESETS:
            result["presets"] = self.presets
        return result

    def get_phase(self, name: str) -> PhaseConfig | None:
        """Get a phase config by name."""
        for p in self.phases:
            if p.phase == name:
                return p
        return None

    def enabled_phases(self) -> list[PhaseConfig]:
        """Get only enabled phases in order."""
        return [p for p in self.phases if p.enabled]

    def build_base_config(self, naked_engine: dict[str, Any]) -> dict[str, Any]:
        """Build initial base_config from naked_engine + global_flags.

        Global flags override naked_engine values. This is the starting
        config that gets passed to the first phase.
        """
        config = dict(naked_engine)
        config.update(self.global_flags)
        return config

    def strip_globals_from_flags(self, test_flags: list[str]) -> list[str]:
        """Remove any flags that are set as global flags from A/B test list."""
        return [f for f in test_flags if f not in self.global_flags]
