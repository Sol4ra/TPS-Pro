# llama-optimizer

Automated multi-phase parameter optimizer for llama-server (llama.cpp) using Optuna-based coordinate descent search.

## Quick Start

**Requirements:** Python 3.9+, a working `llama-server` binary, and a GGUF model file.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the optimizer
python -m tps_pro --server /path/to/llama-server --model /path/to/model.gguf
```

The interactive menu will guide you through setup and optimization. Use `--help` for all CLI options.

### Presets

- `--preset quick` -- Fewer trials, faster results (~50% of normal)
- `--preset normal` -- Default balance of speed and thoroughness
- `--preset thorough` -- More trials for maximum optimization (~150% of normal)

### Batch Mode

Optimize all GGUF files in a directory:

```bash
python -m tps_pro --batch /path/to/models/ --preset normal
```

## Architecture

The optimizer uses a **Pyramid Pipeline** that runs phases in dependency order, where each phase locks its parameters before the next begins. This avoids confounding variables and finds the global optimum in fewer trials than a flat search.

### Pipeline Phases

| Phase | Name | Method | Purpose |
|-------|------|--------|---------|
| 1 | GPU Offload | Grid sweep | Find optimal n_gpu_layers for VRAM boundary |
| 2 | Tensor Split | Grid sweep | Multi-GPU weight distribution (2+ GPUs only) |
| 3 | KV + Context Sweep | Sweep + NIAH | Test KV cache quantization (f16/fp8/fp4) and max context size |
| 4 | Core Engine | Optuna TPE | Multivariate search over threads, batch size, flash attention, polling, and I/O toggles |
| 5 | Speculation | Optuna TPE | N-gram or draft model speculative decoding parameters |
| 6 | Workload Sim | Direct test | Hot-cache TTFT and concurrent user load testing |
| 7 | Quality/Sampling | Optuna TPE | Temperature, top_p, mirostat -- optimized for quality score |

Each phase persists its results to JSON files in the results directory. Interrupted runs resume from the last completed trial.

### Scoring

Trials are scored using a composite formula (`constants/scoring.py`, `SCORE_VERSION=v3`) that blends:

- Generation tokens/second (dominant signal)
- Prompt processing throughput
- Time to first token (TTFT)
- VRAM efficiency

Quality gates (perplexity, KL-divergence, NIAH accuracy) prevent configs that sacrifice output quality for raw speed.

## Configuration

The optimizer reads `config.json` from the working directory (or a path given via `--config`). Key fields:

```json
{
  "server": "/path/to/llama-server",
  "model": "/path/to/model.gguf",
  "port": 8080,
  "preset": "normal",
  "draft_model": "/path/to/draft.gguf",
  "simulate_users": 4,
  "skip_quality": false,
  "pareto": false
}
```

Hardware is auto-detected (GPUs via pynvml, CPU threads via psutil, NUMA topology). Override with the `hardware` key if needed.

## Reading Results

Results are saved to the `results/` directory (or a model-specific subdirectory in batch mode):

| File | Contents |
|------|----------|
| `*_results.json` | Raw phase output (best params, trial data) |
| `command.txt` | Ready-to-run llama-server command with optimized flags |
| `report.html` | Visual summary of all phases |
| `optuna.db` | SQLite database for Optuna dashboard visualization |

### Optuna Dashboard

View trial history and parameter importance interactively:

```bash
python results/_dashboard_launcher.py
# Opens at http://127.0.0.1:8190
```

## Project Structure

```
tps_pro/
├── tests/               # pytest test suite (1271 tests, 70% coverage)
├── tools/               # Utility scripts
└── src/
    └── tps_pro/
        ├── main.py          # Entry point, CLI menu loop
        ├── pipeline.py      # Pipeline orchestrator (run_full_pipeline, batch_optimize)
        ├── state.py         # AppContext singleton, config loading
        ├── errors.py        # Custom exception hierarchy, error strategy docs
        ├── models.py        # GGUF detection, model classification
        ├── hardware.py      # GPU/CPU/NUMA auto-detection
        ├── constants/       # Tuning constants, timeouts, scoring weights
        ├── result_types/    # TypedDict configs, frozen dataclass results
        ├── measurement/     # TPS measurement, scoring, concurrent load
        ├── phases/          # One module per pipeline phase
        ├── engine/          # Server lifecycle (start, kill, wait, bench)
        ├── evals/           # Quality evaluation (MCQ, NIAH, perplexity, KL-div)
        ├── cli/             # Menu, wizard, display, services, report generation
        ├── search/          # Optuna study management, result persistence
        └── data/            # Reference texts, evaluation prompts (JSON)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -q

# Lint
ruff check .

# Type check
mypy --ignore-missing-imports .
```

## License

See repository for license details.
