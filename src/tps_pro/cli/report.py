"""HTML report output."""

from __future__ import annotations

import html as _html
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..engine.util import read_json_safe
from ..hardware import detect_gpus
from ..state import ctx

logger = logging.getLogger(__name__)


_REPORT_PHASE_NAMES = [
    "gpu",
    "tensor_split",
    "topology_sweep",
    "moe_combined",
    "moe",
    "experts",
    "core_engine",
    "io_toggles",
    "speculation",
    "kv_quality",
    "quality",
    "context_sweep",
]


def _load_report_phases(results_dir: Path) -> dict[str, Any]:
    """Load all phase result files for the HTML report.

    Returns:
        dict: Mapping of phase name to parsed result data.
    """
    phases = {}
    for name in _REPORT_PHASE_NAMES:
        path = results_dir / f"{name}_results.json"
        if path.exists():
            data = read_json_safe(path, logger)
            if data is not None:
                phases[name] = data
            else:
                logger.warning("  %s results corrupted — skipping", name)
    return phases


def _build_phase_table_rows(phases: dict[str, Any]) -> str:
    """Build HTML table rows for each phase result.

    Args:
        phases: Dict of phase name to result data.

    Returns:
        str: HTML table row markup.
    """
    rows = ""
    for name, data in phases.items():
        if name in ("context_sweep",):
            continue
        score = data.get("best_tps", data.get("best_score", data.get("best_ngl", "-")))
        dur = data.get("duration_minutes", data.get("duration_seconds", 0))
        if isinstance(dur, (int, float)) and 0 < dur < 1:
            dur_str = f"{dur * 60:.0f}s"
        elif isinstance(dur, (int, float)):
            dur_str = f"{dur:.1f}m" if dur < 60 else f"{dur / 60:.1f}h"  # noqa: PLR2004
        else:
            dur_str = "-"
        trials = len(data.get("all_trials", []))
        beat = data.get("beat_baseline", True)
        color = "#4ade80" if beat else "#f87171"
        rows += f"""
        <tr>
            <td>{_html.escape(str(name))}</td>
            <td style="color:{color}">{_html.escape(str(score))}</td>
            <td>{_html.escape(str(trials))}</td>
            <td>{_html.escape(str(dur_str))}</td>
        </tr>"""
    return rows


def _build_importance_html(phases: dict[str, Any]) -> str:
    """Build HTML for parameter importance bar charts.

    Args:
        phases: Dict of phase name to result data.

    Returns:
        str: HTML markup for importance sections.
    """
    importance_html = ""
    for name in ["core_engine", "io_toggles", "speculation", "kv_quality", "quality"]:
        data = phases.get(name, {})
        imp = data.get("param_importance", {})
        if imp:
            bars = ""
            max_val = max(imp.values()) if imp else 1
            for param, pct in sorted(imp.items(), key=lambda x: -x[1]):
                width = pct / max_val * 100 if max_val > 0 else 0
                bars += f"""
                <div class="imp-row">
                    <span class="imp-name">{_html.escape(str(param))}</span>
                    <div class="imp-bar" style="width:{width:.0f}%"></div>
                    <span class="imp-pct">{pct:.1f}%</span>
                </div>"""
            importance_html += f"<h3>{_html.escape(str(name))}</h3>{bars}"
    return importance_html


def generate_html_report(
    results_dir: str | Path | None = None,
    model_name: str | None = None,
    gpus: list[dict[str, Any]] | None = None,
) -> str | None:
    """Generate a comprehensive HTML report from optimization results."""
    from ..constants import SCORE_PP_BASELINE, SCORE_VERSION, TTFT_BASELINE_MS

    results_dir = Path(results_dir or ctx.results_dir)
    model_name = model_name or ctx.model_path.name
    if gpus is None:
        gpus = detect_gpus()

    phases = _load_report_phases(results_dir)
    if not phases:
        logger.warning("No results found for HTML report.")
        return None

    gpu_info = (
        ", ".join(f"{g['name']} ({g['vram_total_gb']}GB)" for g in gpus)
        if gpus
        else "Unknown"
    )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    phase_rows = _build_phase_table_rows(phases)
    importance_html = _build_importance_html(phases)

    # Get command
    cmd_path = results_dir / "command.txt"
    command_text = cmd_path.read_text() if cmd_path.exists() else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Optimize Report — {_html.escape(model_name)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace; font-size: 14px; padding: 2rem; }}
  h1 {{ color: #00d4ff; margin-bottom: 0.5rem; }}
  h2 {{ color: #00d4ff; margin: 2rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }}
  h3 {{ color: #a0a0a0; margin: 1.5rem 0 0.5rem; }}
  .header {{ margin-bottom: 2rem; }}
  .meta {{ color: #888; margin-bottom: 0.25rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #2a2a4a; }}
  th {{ color: #00d4ff; }}
  .imp-row {{ display: flex; align-items: center; margin: 0.25rem 0; }}
  .imp-name {{ width: 200px; color: #a0a0a0; }}
  .imp-bar {{ height: 16px; background: linear-gradient(90deg, #00d4ff, #0066ff); border-radius: 2px; margin: 0 0.5rem; min-width: 2px; }}
  .imp-pct {{ color: #888; width: 60px; }}
  pre {{ background: #0d0d1a; padding: 1rem; border-radius: 4px; overflow-x: auto; border: 1px solid #333; }}
  .score-formula {{ background: #0d0d1a; padding: 1rem; border-radius: 4px; border: 1px solid #333; margin: 1rem 0; }}
</style>
</head>
<body>
<div class="header">
  <h1>Optimization Report</h1>
  <div class="meta">Model: {_html.escape(model_name)}</div>
  <div class="meta">GPU: {_html.escape(gpu_info)}</div>
  <div class="meta">Generated: {ts}</div>
  <div class="meta">Score Version: {SCORE_VERSION}</div>
</div>

<h2>Phase Summary</h2>
<table>
  <tr><th>Phase</th><th>Best Score</th><th>Trials</th><th>Duration</th></tr>
  {phase_rows}
</table>

<h2>Parameter Importance</h2>
{importance_html if importance_html else "<p>No importance data available.</p>"}

<h2>Score Formula</h2>
<div class="score-formula">
  <p><b>Full mode</b> (promoted configs with large-prompt data):<br>
  score = gen*0.35 + large_tps*0.25 + pp_norm*gen*0.15 + ttft_norm*gen*0.15 + vram_eff*gen*0.10</p>
  <p><b>Lightweight mode</b> (quick filter):<br>
  score = gen * (0.60 + 0.25*pp/{SCORE_PP_BASELINE} + 0.15*{TTFT_BASELINE_MS}/ttft)</p>
  <p style="color:#888;margin-top:0.5rem">Full mode rewards configs that maintain TPS under heavy context load.</p>
</div>

<h2>Optimized Command</h2>
<pre>{_html.escape(command_text) if command_text else "Run generate_optimized_command() to produce this."}</pre>

<footer style="margin-top:3rem;color:#555;text-align:center">
  Generated by llama-server Parameter Optimizer
</footer>
</body>
</html>"""

    report_path = results_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("  HTML report saved to %s", report_path)
    return str(report_path)
