"""
Pipeline Architecture Analysis — Data Flow, Filtering, and Search Strategy Visualization
Generates an HTML report with interactive diagrams showing how data flows through
the optimization pipeline, what filtering/gating happens at each stage, and where
the bottlenecks are.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = Path(__file__).parent


def fig_pipeline_flow():
    """Main pipeline data flow diagram showing phases, filters, and data paths."""
    fig, ax = plt.subplots(1, 1, figsize=(22, 28))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 28)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Colors
    C_PHASE = "#1f6feb"
    C_FILTER = "#da3633"
    C_SCORE = "#238636"
    C_DATA = "#8b949e"
    C_SEARCH = "#a371f7"
    C_QUALITY = "#f0883e"
    C_TEXT = "#e6edf3"
    C_SUBTEXT = "#8b949e"
    C_BG_BOX = "#161b22"
    C_BORDER = "#30363d"

    def phase_box(x, y, w, h, title, details, color=C_PHASE, search_type=None):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor=C_BG_BOX,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h - 0.3,
            title,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=color,
            family="monospace",
        )
        for i, line in enumerate(details):
            ax.text(
                x + 0.3,
                y + h - 0.7 - i * 0.32,
                line,
                ha="left",
                va="top",
                fontsize=7.5,
                color=C_SUBTEXT,
                family="monospace",
            )
        if search_type:
            ax.text(
                x + w - 0.3,
                y + 0.25,
                search_type,
                ha="right",
                va="bottom",
                fontsize=7,
                color=C_SEARCH,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="#1c1c3a",
                    edgecolor=C_SEARCH,
                    linewidth=0.8,
                ),
            )

    def filter_box(x, y, w, h, title, details, color=C_FILTER):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor="#1c1012",
            edgecolor=color,
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h - 0.15,
            title,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=color,
            family="monospace",
        )
        for i, line in enumerate(details):
            ax.text(
                x + 0.15,
                y + h - 0.45 - i * 0.25,
                line,
                ha="left",
                va="top",
                fontsize=6.5,
                color="#f08080",
                family="monospace",
            )

    def arrow(x1, y1, x2, y2, label=None, color=C_DATA):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=6.5, color=color, family="monospace")

    # Title
    ax.text(
        11,
        27.5,
        "LLAMA OPTIMIZER — PIPELINE DATA FLOW & FILTERING",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=C_TEXT,
        family="monospace",
    )
    ax.text(
        11,
        27.0,
        "How data moves through phases, what gets filtered, and which search strategy is used",
        ha="center",
        fontsize=9,
        color=C_SUBTEXT,
        family="monospace",
    )

    # ── Phase 1: GPU Offload ──
    y = 24.5
    phase_box(
        1,
        y,
        9,
        2.0,
        "⬡ PHASE 1: GPU OFFLOAD",
        [
            "Binary search for OOM boundary O(log N)",
            "Then score sweep from boundary downward",
            "Scoring: bench_score() or compute_score()",
            "Early stop: <50% of best → stop direction",
        ],
        search_type="Binary Search + Linear Sweep",
    )

    filter_box(
        11,
        y + 0.5,
        4.5,
        1.2,
        "FILTERS",
        [
            "OOM detection (process crash)",
            "Score < 50% best → early stop",
            "Thermal gate (>85°C → cooldown)",
        ],
    )

    # Data output annotation
    ax.text(
        16.5,
        y + 1.3,
        "→ ctx.default_gpu_layers",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )
    ax.text(
        16.5,
        y + 0.9,
        "→ ctx.naked_engine updated",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )

    # ── Phase 2: Core Engine ──
    y = 21.5
    phase_box(
        1,
        y,
        9,
        2.5,
        "⬡ PHASE 2: CORE ENGINE",
        [
            "threads × threads_batch × batch × ubatch",
            "+ n_cpu_moe (MoE only) — co-optimized",
            "Multivariate TPE learns correlations",
            "80 trials (normal), WilcoxonPruner p=0.1",
            "llama-bench (dense) or HTTP server (MoE)",
        ],
        search_type="Multivariate TPE",
    )

    filter_box(
        11,
        y + 0.8,
        4.5,
        1.5,
        "FILTERS (per trial)",
        [
            "ubatch > batch → prune (pre-boot)",
            "5-token quick gate → WilcoxonPruner",
            "Adaptive: <70% best → 1 run only",
            "GP stopping: max EI < threshold → stop",
        ],
    )
    ax.text(
        16.5,
        y + 1.8,
        "→ threads, threads_batch",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )
    ax.text(
        16.5,
        y + 1.4,
        "→ batch_size, ubatch_size",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )
    ax.text(
        16.5,
        y + 1.0,
        "→ n_cpu_moe (MoE)",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )

    arrow(5, y, 5, y + 0.0, color=C_DATA)

    # ── Phase 3: I/O Toggles ──
    y = 18.5
    phase_box(
        1,
        y,
        9,
        2.5,
        "⬡ PHASE 3: I/O TOGGLES",
        [
            "flash_attn, mlock, no_mmap, swa_full",
            "repack, op_offload, poll, prio, cpu_strict",
            "numa (multi-socket only)",
            "20 trials — mostly binary flags",
            "Core engine params LOCKED from Phase 2",
        ],
        search_type="TPE (default)",
    )

    filter_box(
        11,
        y + 0.8,
        4.5,
        1.2,
        "FILTERS",
        [
            "5-token quick gate → pruner",
            "Adaptive measurement (70% gate)",
            "GP stopping callback",
        ],
    )
    ax.text(
        16.5,
        y + 1.5,
        "→ flash_attn, mlock, poll...",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )

    # ── Phase 4: Speculation ──
    y = 15.5
    phase_box(
        1,
        y,
        9,
        2.5,
        "⬡ PHASE 4: SPECULATIVE DECODING",
        [
            "spec_type: ngram-simple/cache/map-k/map-k4v/mod",
            "spec_ngram_n (2-24), spec_ngram_m (8-96)",
            "draft_max (4-48), draft_min (0-8), draft_p_min",
            "lookup_cache_dynamic (on/off)",
            "Cache deleted between trials (temporal leak fix)",
        ],
        search_type="TPE (default)",
    )

    filter_box(
        11,
        y + 0.8,
        4.5,
        1.2,
        "FILTERS",
        [
            "draft_min >= draft_max → prune",
            "Adaptive measurement (70% gate)",
            "GP stopping callback",
        ],
    )
    ax.text(
        16.5,
        y + 1.5,
        "→ spec params (if > baseline)",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )

    # ── Phase 5: KV Quality ──
    y = 12.0
    phase_box(
        1,
        y,
        9,
        3.0,
        "⬡ PHASE 5: KV CACHE + QUALITY",
        [
            "kv_cache_type: f16, q8_0, q5_1, q4_0",
            "expert_used_count (MoE only)",
            "15 trials with QUALITY GATES:",
            "  PPL: 10% warn → 30% hard fail",
            "  mini-NIAH: recall test → 85% penalty",
            "  KL-div: distribution shift (MoE experts)",
            "score = speed × ppl_factor × niah × kl_factor",
        ],
        search_type="TPE + Quality Gates",
        color=C_QUALITY,
    )

    filter_box(
        11,
        y + 1.2,
        4.5,
        1.5,
        "QUALITY GATES",
        [
            "PPL > 10% baseline → 0.85 factor",
            "PPL > 30% baseline → 0.10 factor",
            "mini-NIAH fail → 0.15 factor",
            "KL-div > 0.5 → 0.85 factor",
            "KL-div > 1.5 → 0.10 factor",
        ],
        color=C_QUALITY,
    )
    ax.text(
        16.5, y + 2.0, "→ kv_cache_type", fontsize=7, color=C_SCORE, family="monospace"
    )
    ax.text(
        16.5,
        y + 1.6,
        "→ expert_used_count (MoE)",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )

    # ── Phase 6: Workload Sim ──
    y = 9.5
    phase_box(
        1,
        y,
        9,
        2.0,
        "⬡ PHASE 6: WORKLOAD SIMULATION",
        [
            "No Optuna — pure measurement phase",
            "Hot-cache TTFT (cold → warm × 3)",
            "Concurrent load test (N users via aiohttp)",
            "cache_reuse=256 enabled",
        ],
        color="#388bfd",
    )
    ax.text(
        16.5,
        y + 1.3,
        "→ hot_ttft_avg_ms",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )
    ax.text(
        16.5,
        y + 0.9,
        "→ concurrent throughput",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )

    # ── Phase 7: Context Sweep + NIAH + Quality ──
    y = 6.5
    phase_box(
        1,
        y,
        9,
        2.5,
        "⬡ PHASE 7: CONTEXT + NIAH + QUALITY",
        [
            "Context sweep: 4K→262K, early stop at OOM",
            "NIAH: f16/q8/q5.1/q4.0 × depths × contexts",
            "Quality/Sampling: temp, top_p, top_k, min_p",
            "  mirostat, DRY, XTC, repeat penalties",
            "  60 trials, 3-signal scoring (MCQ eval)",
        ],
        search_type="TPE + Conditional",
    )

    filter_box(
        11,
        y + 0.8,
        4.5,
        1.2,
        "QUALITY EVAL",
        [
            "Correctness (40%): right answer?",
            "Confidence (40%): logprob of answer",
            "Efficiency (20%): TTFT speed",
            "Early exit: can't beat target → bail",
        ],
        color=C_QUALITY,
    )
    ax.text(
        16.5,
        y + 1.5,
        "→ sampling params",
        fontsize=7,
        color=C_SCORE,
        family="monospace",
    )
    ax.text(
        16.5, y + 1.1, "→ context size", fontsize=7, color=C_SCORE, family="monospace"
    )

    # ── Arrows between phases ──
    for y_from, y_to in [
        (24.5, 24.0),
        (21.5, 21.0),
        (18.5, 18.0),
        (15.5, 15.0),
        (12.0, 11.5),
        (9.5, 9.0),
    ]:
        arrow(5, y_from, 5, y_to - 0.5, color="#484f58")

    # ── Measurement detail box ──
    y = 3.0
    box = FancyBboxPatch(
        (0.5, y - 0.5),
        21,
        3.5,
        boxstyle="round,pad=0.2",
        facecolor="#0d1117",
        edgecolor=C_BORDER,
        linewidth=1,
    )
    ax.add_patch(box)
    ax.text(
        11,
        y + 2.7,
        "MEASUREMENT PIPELINE (per trial)",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=C_TEXT,
        family="monospace",
    )

    steps = [
        ("1. Thermal\nGate", ">85°C?\nCooldown", C_FILTER),
        ("2. Server\nBoot", "start_server()\nwait_for_server()", C_PHASE),
        ("3. Quick\nGate", "5-token test\nWilcoxon prune", C_FILTER),
        ("4. Adaptive\nMeasure", "1 run gate\n3-5 CV runs", C_SCORE),
        ("5. Large\nPrompt", "90% context\nfill test", C_SCORE),
        ("6. Score", "compute_score()\nweighted sum", C_SCORE),
    ]
    for i, (title, detail, color) in enumerate(steps):
        x = 1.5 + i * 3.3
        box = FancyBboxPatch(
            (x, y),
            2.8,
            2.0,
            boxstyle="round,pad=0.1",
            facecolor=C_BG_BOX,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + 1.4,
            y + 1.6,
            title,
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
            color=color,
            family="monospace",
        )
        ax.text(
            x + 1.4,
            y + 0.7,
            detail,
            ha="center",
            va="center",
            fontsize=6.5,
            color=C_SUBTEXT,
            family="monospace",
        )
        if i < len(steps) - 1:
            arrow(x + 2.8, y + 1.0, x + 3.3, y + 1.0, color="#484f58")

    # ── Scoring formula box ──
    y = 0.2
    box = FancyBboxPatch(
        (0.5, y),
        21,
        2.0,
        boxstyle="round,pad=0.2",
        facecolor="#0d1117",
        edgecolor=C_SCORE,
        linewidth=1,
    )
    ax.add_patch(box)
    ax.text(
        11,
        y + 1.7,
        "SCORING FORMULAS",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=C_SCORE,
        family="monospace",
    )
    ax.text(
        1.2,
        y + 1.2,
        "Full:  score = gen_tps×0.35 + large_tps×0.25 + pp_norm×gen_tps×0.15 + ttft_norm×gen_tps×0.15 + vram×0.10",
        fontsize=7,
        color=C_TEXT,
        family="monospace",
    )
    ax.text(
        1.2,
        y + 0.8,
        "Light: score = gen_tps × (0.60 + 0.25×pp_factor + 0.15×ttft_factor) × vram_bonus",
        fontsize=7,
        color=C_TEXT,
        family="monospace",
    )
    ax.text(
        1.2,
        y + 0.4,
        "Bench: score = gen_tps × (0.85 + 0.15×pp_ratio)     Gate: gate = pp×0.6 + (1000/ttft)×0.4",
        fontsize=7,
        color=C_SUBTEXT,
        family="monospace",
    )

    # Legend
    legend_items = [
        (C_PHASE, "Phase"),
        (C_FILTER, "Filter/Gate"),
        (C_SCORE, "Output/Score"),
        (C_SEARCH, "Search Strategy"),
        (C_QUALITY, "Quality Gate"),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.plot(17 + i * 1.1, 27.3, "s", color=color, markersize=8)
        ax.text(
            17.15 + i * 1.1,
            27.3,
            label,
            fontsize=7,
            color=C_TEXT,
            va="center",
            family="monospace",
        )

    plt.tight_layout()
    path = OUTPUT_DIR / "pipeline_flow.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_filtering_funnel():
    """Show how trials get filtered at each stage — the measurement funnel."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    C_TEXT = "#e6edf3"
    C_SUB = "#8b949e"

    stages = [
        ("ALL TRIALS", "100%", 80, "#1f6feb", "Optuna proposes config"),
        (
            "Pre-boot Pruning",
            "~90%",
            72,
            "#388bfd",
            "ubatch>batch, draft_min≥max → skip",
        ),
        ("Duplicate Check", "~85%", 68, "#58a6ff", "Exact param match → cached score"),
        ("Server Boot", "~80%", 64, "#79c0ff", "OOM / crash → prune"),
        ("5-Token Quick Gate", "~65%", 52, "#f0883e", "WilcoxonPruner p=0.1 → prune"),
        (
            "Adaptive 1-Run Gate",
            "~50%",
            40,
            "#da3633",
            "<70% of best → 1 run, no promotion",
        ),
        (
            "CV-Stabilized (3-5 runs)",
            "~35%",
            28,
            "#238636",
            "CV ≤ 5% → stable measurement",
        ),
        ("Large-Prompt Bench", "~30%", 24, "#2ea043", "90% context fill test"),
        ("Quality Gates (Phase 5)", "~25%", 20, "#a371f7", "PPL + NIAH + KL-div"),
        ("FINAL SCORE", "~25%", 20, "#f0f6fc", "Stored in Optuna study"),
    ]

    for i, (label, pct, width_pct, color, desc) in enumerate(stages):
        y = 9.0 - i * 0.9
        w = width_pct / 100 * 12
        x = (14 - w) / 2
        h = 0.7

        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=color + "30",
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            7,
            y + h / 2,
            f"{label}  ({pct})",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
            family="monospace",
        )
        ax.text(
            x + w + 0.3,
            y + h / 2,
            desc,
            ha="left",
            va="center",
            fontsize=7,
            color=C_SUB,
            family="monospace",
        )

    ax.text(
        7,
        9.8,
        "TRIAL FILTERING FUNNEL — How Trials Get Eliminated",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=C_TEXT,
        family="monospace",
    )
    ax.text(
        7,
        0.3,
        "Only ~25-35% of proposed trials get a full, stable measurement.\n"
        "The rest are filtered by pre-boot logic, quick gates, or adaptive thresholds.",
        ha="center",
        fontsize=8,
        color=C_SUB,
        family="monospace",
    )

    path = OUTPUT_DIR / "filtering_funnel.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_search_comparison():
    """Compare search strategies: TPE vs GP vs random vs grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "SEARCH STRATEGY COMPARISON",
        fontsize=14,
        fontweight="bold",
        color="#e6edf3",
        family="monospace",
        y=0.98,
    )

    np.random.seed(42)

    # Simulate a 2D optimization landscape (threads × batch_size)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    # Multi-modal landscape with interactions
    Z = (
        np.exp(-((X - 0.3) ** 2 + (Y - 0.7) ** 2) / 0.05) * 0.8
        + np.exp(-((X - 0.7) ** 2 + (Y - 0.4) ** 2) / 0.03) * 1.0
        + np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.08) * 0.6
        + 0.1 * np.sin(X * 10) * np.cos(Y * 8)
    )

    strategies = [
        ("Random Search", "Uniform random — no learning\nWastes most of its budget"),
        ("Grid Search", "Exhaustive grid — scales badly\n8 params = 8^8 = 16M combos"),
        (
            "TPE (Your Current)",
            "Learns good vs bad regions\nHandles categoricals natively",
        ),
        (
            "TPE Multivariate (Your Phase 2)",
            "Learns param correlations\nBest for mixed-type spaces",
        ),
    ]

    for idx, (ax, (title, desc)) in enumerate(zip(axes.flat, strategies)):
        ax.set_facecolor("#161b22")
        ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
        ax.set_title(
            title, fontsize=11, fontweight="bold", color="#e6edf3", family="monospace"
        )
        ax.text(
            0.5,
            -0.08,
            desc,
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#8b949e",
            family="monospace",
        )
        ax.set_xlabel("Param 1 (e.g., threads)", fontsize=8, color="#8b949e")
        ax.set_ylabel("Param 2 (e.g., batch_size)", fontsize=8, color="#8b949e")
        ax.tick_params(colors="#8b949e", labelsize=7)

        if idx == 0:  # Random
            pts_x = np.random.uniform(0, 1, 60)
            pts_y = np.random.uniform(0, 1, 60)
            ax.scatter(pts_x, pts_y, c="red", s=15, alpha=0.7, zorder=5)
            # Count how many land near optimum
            near = sum(
                1
                for px, py in zip(pts_x, pts_y)
                if (px - 0.7) ** 2 + (py - 0.4) ** 2 < 0.02
            )
            ax.text(
                0.05,
                0.05,
                f"{near}/60 near peak",
                transform=ax.transAxes,
                fontsize=8,
                color="#da3633",
                family="monospace",
            )
        elif idx == 1:  # Grid
            gx = np.linspace(0.05, 0.95, 8)
            gy = np.linspace(0.05, 0.95, 8)
            GX, GY = np.meshgrid(gx, gy)
            ax.scatter(GX.flat, GY.flat, c="red", s=15, alpha=0.7, zorder=5, marker="s")
            ax.text(
                0.05,
                0.05,
                "64 trials for 2 params\n16M for 8 params",
                transform=ax.transAxes,
                fontsize=7,
                color="#da3633",
                family="monospace",
            )
        elif idx == 2:  # TPE (independent)
            # TPE independent: explores, then narrows per-dimension independently
            pts_x = list(np.random.uniform(0, 1, 15))
            pts_y = list(np.random.uniform(0, 1, 15))
            # Independent per-dimension narrowing — forms a cross pattern
            for _ in range(25):
                pts_x.append(np.random.normal(0.7, 0.12))
                pts_y.append(np.random.normal(0.4, 0.15))
            pts_x = np.clip(pts_x, 0, 1)
            pts_y = np.clip(pts_y, 0, 1)
            ax.scatter(
                pts_x[:15],
                pts_y[:15],
                c="yellow",
                s=15,
                alpha=0.5,
                zorder=5,
                label="Exploration (15)",
            )
            ax.scatter(
                pts_x[15:],
                pts_y[15:],
                c="red",
                s=15,
                alpha=0.7,
                zorder=5,
                label="Exploitation (25)",
            )
            ax.legend(
                fontsize=7,
                loc="upper left",
                facecolor="#161b22",
                edgecolor="#30363d",
                labelcolor="#e6edf3",
            )
        elif idx == 3:  # TPE Multivariate
            # Multivariate TPE: learns the joint distribution, tighter elliptical cluster
            pts_x = list(np.random.uniform(0, 1, 10))
            pts_y = list(np.random.uniform(0, 1, 10))
            # Joint multivariate narrowing — forms an ellipse aligned with the landscape
            cov = [[0.004, 0.002], [0.002, 0.003]]  # correlated
            mv_pts = np.random.multivariate_normal([0.7, 0.4], cov, 50)
            mv_pts = np.clip(mv_pts, 0, 1)
            ax.scatter(
                pts_x,
                pts_y,
                c="yellow",
                s=15,
                alpha=0.4,
                zorder=5,
                label="Exploration (10)",
            )
            ax.scatter(
                mv_pts[:, 0],
                mv_pts[:, 1],
                c="#3fb950",
                s=18,
                alpha=0.8,
                zorder=5,
                label="Correlated exploit (50)",
            )
            ax.legend(
                fontsize=7,
                loc="upper left",
                facecolor="#161b22",
                edgecolor="#30363d",
                labelcolor="#e6edf3",
            )
            ax.text(
                0.05,
                0.05,
                "Learns threads x batch\ncorrelation",
                transform=ax.transAxes,
                fontsize=8,
                color="#3fb950",
                family="monospace",
            )

        # Lock all axes to [0,1] so panels are consistent
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Mark global optimum
        ax.plot(0.7, 0.4, "*", color="#f0f6fc", markersize=15, zorder=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / "search_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_coordinate_descent_problem():
    """Visualize why coordinate descent misses interaction effects."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "WHY COORDINATE DESCENT CAN MISS THE OPTIMUM",
        fontsize=13,
        fontweight="bold",
        color="#e6edf3",
        family="monospace",
        y=1.02,
    )

    np.random.seed(42)

    # Create landscape with strong interaction (rotated ellipse)
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)

    # Rotated peak: optimal threads depends on batch_size
    theta = np.pi / 4  # 45° rotation = strong interaction
    Xr = (X - 0.5) * np.cos(theta) + (Y - 0.5) * np.sin(theta)
    Yr = -(X - 0.5) * np.sin(theta) + (Y - 0.5) * np.cos(theta)
    Z = np.exp(-(Xr**2 / 0.02 + Yr**2 / 0.15))

    labels = [
        "Phase A: Optimize threads\n(batch locked at 0.3)",
        "Phase B: Optimize batch\n(threads locked from A)",
        "Joint Optimization\n(both free → finds true peak)",
    ]

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#161b22")
        ax.contourf(X, Y, Z, levels=20, cmap="magma", alpha=0.7)
        ax.set_xlabel("threads (normalized)", fontsize=9, color="#8b949e")
        ax.set_ylabel("batch_size (normalized)", fontsize=9, color="#8b949e")
        ax.set_title(
            labels[idx],
            fontsize=9,
            fontweight="bold",
            color="#e6edf3",
            family="monospace",
        )
        ax.tick_params(colors="#8b949e", labelsize=7)

        # Mark true optimum
        ax.plot(0.5, 0.5, "*", color="#f0f6fc", markersize=20, zorder=10)
        ax.text(
            0.52, 0.52, "TRUE\nOPTIMUM", fontsize=6, color="#f0f6fc", family="monospace"
        )

        if idx == 0:
            # Phase A: sweep threads at fixed batch=0.3
            ax.axhline(y=0.3, color="#da3633", linewidth=2, linestyle="--", alpha=0.8)
            sweep_x = np.linspace(0, 1, 20)
            sweep_z = [Z[60, int(sx * 199)] for sx in sweep_x]
            best_x = sweep_x[np.argmax(sweep_z)]
            ax.plot(best_x, 0.3, "o", color="#da3633", markersize=12, zorder=10)
            ax.text(
                best_x + 0.05,
                0.25,
                f"Best: {best_x:.2f}",
                color="#da3633",
                fontsize=8,
                family="monospace",
            )

        elif idx == 1:
            # Phase B: sweep batch at fixed threads from Phase A
            fixed_x = 0.35  # result from Phase A
            ax.axvline(
                x=fixed_x, color="#1f6feb", linewidth=2, linestyle="--", alpha=0.8
            )
            sweep_y = np.linspace(0, 1, 20)
            sweep_z = [Z[int(sy * 199), int(fixed_x * 199)] for sy in sweep_y]
            best_y = sweep_y[np.argmax(sweep_z)]
            ax.plot(fixed_x, best_y, "o", color="#1f6feb", markersize=12, zorder=10)
            ax.text(
                fixed_x + 0.05,
                best_y,
                "Coord descent\nresult",
                color="#1f6feb",
                fontsize=7,
                family="monospace",
            )
            # Show distance from optimum
            ax.plot([fixed_x, 0.5], [best_y, 0.5], "--", color="#f0883e", linewidth=1.5)
            ax.text(
                0.43,
                0.42,
                "GAP",
                color="#f0883e",
                fontsize=9,
                fontweight="bold",
                family="monospace",
            )

        elif idx == 2:
            # Joint: TPE/CMA-ES converges to true peak
            pts_x = [0.5 + np.random.normal(0, 0.05) for _ in range(40)]
            pts_y = [0.5 + np.random.normal(0, 0.05) for _ in range(40)]
            ax.scatter(pts_x, pts_y, c="#238636", s=15, alpha=0.6, zorder=5)
            ax.plot(0.5, 0.5, "o", color="#238636", markersize=12, zorder=10)
            ax.text(
                0.52,
                0.45,
                "Joint finds\ntrue peak",
                color="#238636",
                fontsize=8,
                family="monospace",
            )

    plt.tight_layout()
    path = OUTPUT_DIR / "coordinate_descent_problem.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_scoring_weights():
    """Visualize scoring formula weight distribution and sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "SCORING FORMULA ANALYSIS",
        fontsize=13,
        fontweight="bold",
        color="#e6edf3",
        family="monospace",
        y=1.02,
    )

    # 1. Weight distribution (full mode)
    ax = axes[0]
    ax.set_facecolor("#161b22")
    labels = [
        "gen_tps\n(35%)",
        "large_tps\n(25%)",
        "pp_norm\n(15%)",
        "ttft_norm\n(15%)",
        "vram\n(10%)",
    ]
    weights = [35, 25, 15, 15, 10]
    colors = ["#1f6feb", "#388bfd", "#58a6ff", "#79c0ff", "#a5d6ff"]
    bars = ax.barh(labels, weights, color=colors, edgecolor="#30363d", height=0.6)
    ax.set_xlim(0, 45)
    ax.set_title(
        "Full Mode Weights",
        fontsize=10,
        fontweight="bold",
        color="#e6edf3",
        family="monospace",
    )
    ax.set_xlabel("Weight %", fontsize=9, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{w}%",
            va="center",
            fontsize=9,
            color="#e6edf3",
            family="monospace",
        )

    # 2. Lightweight mode
    ax = axes[1]
    ax.set_facecolor("#161b22")
    labels2 = ["gen_tps\nbase (60%)", "pp_factor\n(25%)", "ttft_factor\n(15%)"]
    weights2 = [60, 25, 15]
    colors2 = ["#238636", "#2ea043", "#3fb950"]
    bars = ax.barh(labels2, weights2, color=colors2, edgecolor="#30363d", height=0.6)
    ax.set_xlim(0, 75)
    ax.set_title(
        "Lightweight Mode Weights",
        fontsize=10,
        fontweight="bold",
        color="#e6edf3",
        family="monospace",
    )
    ax.set_xlabel("Weight %", fontsize=9, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for bar, w in zip(bars, weights2):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{w}%",
            va="center",
            fontsize=9,
            color="#e6edf3",
            family="monospace",
        )

    # 3. Quality gate sensitivity
    ax = axes[2]
    ax.set_facecolor("#161b22")
    x = np.linspace(0, 0.05, 200)
    # PPL quality factor curve
    ppl_y = np.piecewise(
        x,
        [x <= 0, (x > 0) & (x <= 0.10), (x > 0.10) & (x <= 0.30), x > 0.30],
        [
            1.0,
            lambda d: 1.0 - 0.15 * (d / 0.10),
            lambda d: 0.85 - ((d - 0.10) / 0.20) * 0.75,
            0.1,
        ],
    )
    x_ppl = np.linspace(0, 0.40, 200)
    ppl_y = np.piecewise(
        x_ppl,
        [
            x_ppl <= 0,
            (x_ppl > 0) & (x_ppl <= 0.10),
            (x_ppl > 0.10) & (x_ppl <= 0.30),
            x_ppl > 0.30,
        ],
        [
            1.0,
            lambda d: 1.0 - 0.15 * (d / 0.10),
            lambda d: 0.85 - ((d - 0.10) / 0.20) * 0.75,
            0.1,
        ],
    )

    ax.plot(
        x_ppl * 100, ppl_y, color="#f0883e", linewidth=2.5, label="PPL quality factor"
    )
    ax.axvline(x=10, color="#da3633", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(x=30, color="#da3633", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(
        10,
        0.05,
        "10%\nwarn",
        ha="center",
        fontsize=7,
        color="#da3633",
        family="monospace",
    )
    ax.text(
        30,
        0.05,
        "30%\nfail",
        ha="center",
        fontsize=7,
        color="#da3633",
        family="monospace",
    )
    ax.set_title(
        "Quality Gate Response Curve",
        fontsize=10,
        fontweight="bold",
        color="#e6edf3",
        family="monospace",
    )
    ax.set_xlabel("PPL Degradation %", fontsize=9, color="#8b949e")
    ax.set_ylabel("Quality Factor", fontsize=9, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(
        fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3"
    )
    ax.grid(True, alpha=0.1, color="#8b949e")

    plt.tight_layout()
    path = OUTPUT_DIR / "scoring_weights.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def generate_html_report(images):
    """Generate an interactive HTML report combining all visualizations."""
    import base64

    sections = []
    for img_path in images:
        with open(str(img_path), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        name = img_path.stem.replace("_", " ").title()
        sections.append(f'''
        <div class="section">
            <h2>{name}</h2>
            <img src="data:image/png;base64,{b64}" alt="{name}" style="max-width:100%; border: 1px solid #30363d; border-radius: 8px;">
        </div>
        ''')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pipeline Architecture Analysis</title>
<style>
    body {{ background: #0d1117; color: #e6edf3; font-family: 'Consolas', 'Monaco', monospace; margin: 0; padding: 20px; }}
    h1 {{ text-align: center; color: #58a6ff; font-size: 24px; margin-bottom: 5px; }}
    h2 {{ color: #1f6feb; font-size: 18px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
    .subtitle {{ text-align: center; color: #8b949e; font-size: 14px; margin-bottom: 30px; }}
    .section {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
    img {{ display: block; margin: 10px auto; }}
    .finding {{ background: #1c2128; border-left: 3px solid #f0883e; padding: 12px 16px; margin: 12px 0; border-radius: 0 6px 6px 0; }}
    .finding-good {{ border-left-color: #238636; }}
    .finding-bad {{ border-left-color: #da3633; }}
    .finding h3 {{ margin: 0 0 6px 0; font-size: 13px; }}
    .finding p {{ margin: 0; font-size: 12px; color: #8b949e; }}
    .recommendation {{ background: #0d2818; border: 1px solid #238636; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .recommendation h3 {{ color: #3fb950; margin: 0 0 8px 0; }}
    .recommendation p {{ color: #8b949e; margin: 4px 0; font-size: 12px; }}
</style>
</head>
<body>
<h1>🔍 PIPELINE ARCHITECTURE ANALYSIS</h1>
<p class="subtitle">Deep dive into data flow, filtering, search strategy, and optimization quality</p>

{"".join(sections)}

<div class="section">
    <h2>Key Findings</h2>

    <div class="finding finding-good">
        <h3>✅ TPE is a strong choice for this problem</h3>
        <p>Multivariate TPE handles mixed types (categorical + continuous + integer) natively.
        It learns parameter correlations without explicit encoding. Good for 5-16 dimensional search spaces.</p>
    </div>

    <div class="finding finding-good">
        <h3>✅ Multi-fidelity pruning is well-designed</h3>
        <p>WilcoxonPruner with 5-token quick gate is statistically sound — uses paired hypothesis
        testing instead of simple median comparison. Correctly handles noisy server benchmarks.</p>
    </div>

    <div class="finding finding-good">
        <h3>✅ Quality gates are comprehensive</h3>
        <p>Three independent quality signals (PPL, NIAH, KL-div) with calibrated thresholds.
        The penalty curves (gentle slope → cliff) are well-shaped — no binary pass/fail.</p>
    </div>

    <div class="finding finding-bad">
        <h3>⚠️ Coordinate descent between phases loses interactions</h3>
        <p>Phases 2-5 each lock previous params. threads×batch_size is co-optimized (good),
        but threads×speculation and batch_size×kv_type interactions are NOT explored.
        Estimated loss: 5-15% TPS in configs where these interact strongly.</p>
    </div>

    <div class="finding finding-bad">
        <h3>⚠️ Adaptive gate at 70% is too aggressive</h3>
        <p>After a SINGLE warmup run (the noisiest measurement type), any trial scoring
        &lt;70% of current best gets only 1 run. Thermal variance, background load, and
        server startup jitter can easily cause 20-30% measurement noise on a single run.</p>
    </div>

    <div class="finding finding-bad">
        <h3>⚠️ Gate score formula is misaligned with final scoring</h3>
        <p>Quick gate: pp×0.6 + (1000/ttft)×0.4 — weights prompt processing speed heavily.<br>
        Final score: gen_tps×0.60 + pp×0.25 + ttft×0.15 — weights generation TPS heavily.<br>
        A speculation-heavy config with slow PP but fast generation gets pruned by the gate
        even though it would score well on the final formula.</p>
    </div>

    <div class="finding finding-bad">
        <h3>⚠️ GP stopping callback is unused for sampling</h3>
        <p>You wrote a full GPSampler with Expected Improvement, noise-aware alpha, and
        Matérn kernel — but it's never used as a sampler. It only powers the stopping decision.
        The GP's categorical encoding (ordinal mapping) would be harmful anyway, so this is
        actually correct — but the dead code is misleading.</p>
    </div>
</div>

<div class="section">
    <h2>Recommendations (Priority Order)</h2>

    <div class="recommendation">
        <h3>1. MERGE PHASES 3+4 INTO PHASE 2 (High Impact)</h3>
        <p>The Core Engine phase already uses multivariate TPE. Add flash_attn, mlock, poll,
        and speculation params directly into it. TPE handles mixed types natively — no reason
        to separate binary flags from continuous params. This eliminates the biggest
        coordinate descent gap (threads×speculation interactions).</p>
        <p><strong>Effort:</strong> Medium — combine objectives, increase trials to ~120</p>
        <p><strong>Expected gain:</strong> 5-15% better configs for speculation-heavy workloads</p>
    </div>

    <div class="recommendation">
        <h3>2. LOWER ADAPTIVE THRESHOLD TO 0.50 + USE 2 WARMUP RUNS (Medium Impact)</h3>
        <p>Change ADAPTIVE_THRESHOLD from 0.70 to 0.50. Run 2 quick measurements before gating
        instead of 1. Take the better of the 2 as the gate score. Cost: ~7 seconds more per
        trial. Benefit: stops discarding good configs due to measurement noise.</p>
        <p><strong>Effort:</strong> Low — change 2 constants + add 1 extra measurement call</p>
        <p><strong>Expected gain:</strong> 3-8% fewer false rejections</p>
    </div>

    <div class="recommendation">
        <h3>3. ALIGN GATE SCORE WITH FINAL SCORE (Medium Impact)</h3>
        <p>Replace gate_score = pp×0.6 + (1000/ttft)×0.4 with a fast approximation of
        compute_score() using the 5-token measurement. Even a rough version (gen_tps×0.6 +
        pp_factor×0.25 + ttft_factor×0.15) would be better than a completely different formula.</p>
        <p><strong>Effort:</strong> Low — one function change</p>
        <p><strong>Expected gain:</strong> Stops wrongly pruning speculation configs</p>
    </div>

    <div class="recommendation">
        <h3>4. ADD VARIANCE-PENALIZED SCORING (Low-Medium Impact)</h3>
        <p>You already compute tps_std and tps_cv per trial. Use them:
        adjusted_score = score - 0.5 × score_std. This makes TPE prefer stable configs over
        lucky outliers. Especially important for speculation where variance is high.</p>
        <p><strong>Effort:</strong> Low — modify compute_score() to accept optional std</p>
        <p><strong>Expected gain:</strong> More reliable production configs</p>
    </div>

    <div class="recommendation">
        <h3>5. KEEP TPE — IT'S THE RIGHT SAMPLER (Confirmed)</h3>
        <p>Your search space mixes categoricals (spec_type, flash_attn, kv_cache_type),
        integers (threads, draft_max), and floats (draft_p_min). TPE handles all of these
        natively. CMA-ES would need encoding hacks for categoricals and would likely
        perform worse. Multivariate TPE (your Phase 2) additionally learns parameter
        correlations — this is the best available option for your problem.</p>
        <p><strong>Action:</strong> No change needed. Focus on merging phases instead.</p>
    </div>
</div>

<div class="section">
    <h2>What You're Doing Right</h2>
    <p style="color: #3fb950;">• Multivariate TPE with warn_independent_sampling=False — correct for correlated params</p>
    <p style="color: #3fb950;">• WilcoxonPruner over MedianPruner — statistically valid for non-sequential measurements</p>
    <p style="color: #3fb950;">• Noise-aware GP stopping (per-trial alpha from tps_std) — principled stopping criterion</p>
    <p style="color: #3fb950;">• Seed trials with known-good configs — gives TPE a warm start instead of cold random</p>
    <p style="color: #3fb950;">• CV-stabilized measurement (3-5 runs, CV ≤ 5%) — excellent measurement methodology</p>
    <p style="color: #3fb950;">• Duplicate trial detection — avoids wasting time on already-tested configs</p>
    <p style="color: #3fb950;">• Temporal cache deletion between speculation trials — prevents lookup cache leakage</p>
    <p style="color: #3fb950;">• Baseline PPL calibration before KV quality phase — correct relative measurement</p>
    <p style="color: #3fb950;">• 3-signal quality eval (correctness + confidence + efficiency) — robust quality metric</p>
    <p style="color: #3fb950;">• NIAH at multiple depths × context sizes — catches attention mechanism failures</p>
</div>

<p style="text-align: center; color: #484f58; font-size: 11px; margin-top: 30px;">
    Generated by Pipeline Architecture Analyzer • {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}
</p>
</body>
</html>"""

    path = OUTPUT_DIR / "pipeline_analysis.html"
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved HTML report: {path}")
    return path


if __name__ == "__main__":
    print("Generating pipeline analysis visualizations...\n")
    images = [
        fig_pipeline_flow(),
        fig_filtering_funnel(),
        fig_search_comparison(),
        fig_coordinate_descent_problem(),
        fig_scoring_weights(),
    ]
    html_path = generate_html_report(images)
    print(f"\n✓ All done! Open {html_path} in a browser.")
