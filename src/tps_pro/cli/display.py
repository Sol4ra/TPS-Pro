"""Result viewing and browsing -- model list, phase results, phase detail.

All data retrieval is delegated to services.py. This module is pure UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..engine.util import read_json_safe
from ..state import ctx
from .services import (
    generate_optimized_command,
    get_model_results,
    get_phase_detail,
    get_phase_results,
)
from .services_pipeline import ModelResultSummary

logger = logging.getLogger(__name__)

SEP = "=" * 60


def view_results() -> None:
    """Browse saved results organized by model."""
    try:
        _migrate_legacy_results()

        model_list = get_model_results(ctx)
        if not model_list:
            print("\n  No optimization results found.")
            return

        _model_selection_loop(model_list)
    except EOFError:
        return
    except Exception as exc:
        print(f"  [!] Error viewing results: {exc}")


def _model_selection_loop(model_list: list[ModelResultSummary]) -> None:
    """Let user pick a model to view its phase results."""
    from .menu import clear_screen

    while True:
        clear_screen()
        print(f"{SEP}")
        print("  Saved Results -- Select Model")
        print(SEP)

        for i, m in enumerate(model_list):
            tps = f" | Best: {m['best_tps']:.1f} t/s" if m["best_tps"] > 0 else ""
            print(f"  [{i + 1}] {m['name']}")
            print(f"      {m['phase_count']} phases | {m['last_modified'][:16]}{tps}")
        print("\n  [b] Back")

        try:
            choice = input("\n  > ").strip().lower()
        except EOFError:
            return

        if choice in ("b", ""):
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_list):
                _show_model_results(Path(model_list[idx]["path"]))
            else:
                print("  Invalid choice.")
        except ValueError:
            print("  Invalid choice.")


def _show_model_results(model_dir: Path) -> None:
    """Show phase results for a model, plus the launch command."""
    try:
        label = model_dir.name

        from .menu import clear_screen

        while True:
            clear_screen()
            phases = get_phase_results(model_dir)
            if not phases:
                print(f"  No phase results in {label}.")
                return

            _print_phase_list(label, phases)
            _print_launch_command(model_dir)

            try:
                choice = input("\n  > ").strip().lower()
            except EOFError:
                return

            if choice in ("b", ""):
                return

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(phases):
                    path = model_dir / f"{phases[idx].name}_results.json"
                    _show_phase_detail(phases[idx].name, path)
                else:
                    print("  Invalid choice.")
            except ValueError:
                print("  Invalid choice.")
    except EOFError:
        return
    except Exception as exc:
        print(f"  [!] Error: {exc}")


def _print_phase_list(label: str, phases: list) -> None:
    """Print numbered list of phase results."""
    print(f"\n{SEP}")
    print(f"  {label} -- Phase Results")
    print(SEP)

    for i, pr in enumerate(phases):
        if not pr.data:
            print(f"  [{i + 1}]   {pr.name} (error reading)")
            continue

        tps = pr.best_tps or 0
        dur_min = (pr.duration_seconds / 60) if pr.duration_seconds else 0
        marker = "+" if pr.beat_baseline else "="
        parts = []
        if tps > 0:
            parts.append(f"{tps:.1f} t/s")
        if dur_min > 0:
            parts.append(f"{dur_min:.1f}m")
        detail = f" ({', '.join(parts)})" if parts else ""
        print(f"  [{i + 1}] {marker} {pr.name}{detail}")

    print("\n  [b] Back")


def _print_launch_command(model_dir: Path) -> None:
    """Print the generated launch command for this model's results."""
    try:
        if str(ctx.results_dir) == str(model_dir):
            cmd = generate_optimized_command(ctx)
        else:
            cmd_path = model_dir / "command.txt"
            if cmd_path.exists():
                cmd = cmd_path.read_text(encoding="utf-8").strip()
            else:
                cmd = None

        if cmd:
            print(f"\n  {'-' * 50}")
            print("  Launch Command:")
            print()
            for line in cmd.splitlines():
                print(f"    {line}")
    except Exception as exc:
        logger.warning("Could not generate launch command: %s", exc)


def _show_phase_detail(phase_name: str, path: Path) -> None:
    """Show detailed results for a single phase."""
    from .menu import clear_screen

    try:
        clear_screen()
        data = get_phase_detail(path)
        if not data:
            print(f"  Error reading {path}")
            return

        print(f"{SEP}")
        print(f"  {phase_name} -- Details")
        print(SEP)

        _print_baseline(data)
        _print_best(data)
        _print_meta(data)
        _print_best_params(data)
        _print_importance(data)
        _print_context_results(data)
        _print_kv_results(data)

        print()
        try:
            input("  Press Enter to continue...")
        except EOFError:
            pass  # User pressed Ctrl+D at prompt — expected in CLI
    except EOFError:
        return
    except Exception as exc:
        logger.warning("Error showing phase detail %s: %s", phase_name, exc)
        print(f"  [!] Error: {exc}")


def _print_baseline(data: dict) -> None:
    """Print baseline metrics if available."""
    bl = data.get("baseline", {})
    if bl:
        print(
            f"  Baseline:  {bl.get('tps', 0):.1f} t/s"
            f" | pp: {bl.get('prompt_tps', 0):.0f} t/s"
            f" | TTFT: {bl.get('ttft', 0):.0f}ms"
        )
    bs = data.get("baseline_score")
    if bs:
        print(f"  Baseline Score: {bs:.1f}")


def _print_best(data: dict) -> None:
    """Print best metrics if available."""
    bm = data.get("best_metrics", {})
    if bm:
        print(
            f"  Best:      {bm.get('tps', 0):.1f} t/s"
            f" | pp: {bm.get('prompt_tps', 0):.0f} t/s"
            f" | TTFT: {bm.get('ttft', 0):.0f}ms"
        )
    elif data.get("best_tps"):
        print(f"  Best TPS:  {data['best_tps']:.1f} t/s")

    if data.get("best_score"):
        print(f"  Best Score: {data['best_score']:.1f}")

    beat = data.get("beat_baseline")
    if beat is not None:
        print(f"  Beat baseline: {'Yes' if beat else 'No'}")

    vf = data.get("verified", {})
    if vf:
        print(
            f"  Verified:  {vf.get('tps', 0):.1f} t/s"
            f" | pp: {vf.get('prompt_tps', 0):.0f} t/s"
            f" | TTFT: {vf.get('ttft', 0):.0f}ms"
        )


def _print_meta(data: dict) -> None:
    """Print duration and trial count."""
    dur = data.get("duration_minutes")
    if dur:
        print(f"  Duration:  {dur:.1f} min")
    trials = data.get("all_trials", [])
    if trials:
        print(f"  Trials:    {len(trials)}")


def _print_best_params(data: dict) -> None:
    """Print best parameters table."""
    bp = data.get("best_params", {})
    if not bp:
        return
    print("\n  Best Params:")
    for k, v in bp.items():
        print(f"    {k}: {v}")


def _print_importance(data: dict) -> None:
    """Print parameter importance bars."""
    pi = data.get("param_importance", {})
    if not pi:
        return
    print("\n  Parameter Importance:")
    for k, v in sorted(pi.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(v / 5)
        print(f"    {k:28s} {v:5.1f}%  {bar}")


def _print_context_results(data: dict) -> None:
    """Print context sweep results if present."""
    contexts = data.get("contexts", {})
    if not contexts:
        return
    print("\n  Context Results:")
    for ctx_str, info in sorted(contexts.items(), key=lambda x: int(x[0])):
        tps = info.get("tps", 0)
        score = info.get("score", 0)
        fits = "yes" if info.get("fits", True) else "OOM"
        print(f"    {int(ctx_str):>8,}: {tps:.1f} t/s | score: {score:.1f} | {fits}")


def _print_kv_results(data: dict) -> None:
    """Print NIAH KV cache results if present."""
    kv = data.get("kv_results", [])
    if not kv:
        return
    print("\n  KV Cache Results:")
    for r in kv:
        ppl = f" | PPL: {r['ppl']:.2f}" if r.get("ppl") else ""
        print(f"    {r['kv_type']:>6}: {r['pass_rate']:.0f}% recall{ppl}")


def _migrate_legacy_results() -> None:
    """Move legacy flat result files into a model subfolder."""
    base = Path(__file__).resolve().parent.parent / "results"
    if not base.exists():
        return

    root_files = list(base.glob("*_results.json"))
    if not root_files:
        return

    cfg_path = base / "optimizer-config.json"
    label = "unknown-model"
    cfg = read_json_safe(cfg_path) if cfg_path.exists() else None
    if cfg:
        mp = cfg.get("model", "")
        if mp:
            label = Path(mp).stem.lower().replace(" ", "-")

    dest = base / label
    dest.mkdir(parents=True, exist_ok=True)
    for f in root_files:
        target = dest / f.name
        if not target.exists():
            f.rename(target)
        else:
            f.unlink()

    root_db = base / "optuna.db"
    if root_db.exists() and not (dest / "optuna.db").exists():
        root_db.rename(dest / "optuna.db")
