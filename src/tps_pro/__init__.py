"""
tps_pro — llama-server Parameter Optimizer
==========================================
Multi-phase coordinate descent using Optuna (GP-Bayesian/TPE).

Usage:
  python -m tps_pro              # use defaults
  python -m tps_pro --model /path/to/model.gguf
  python -m tps_pro --server /path/to/llama-server --port 8091

Error strategy:
    See errors.py for full documentation.  In summary: phase errors are caught by
    pipeline._run_phase() via _PHASE_ERRORS so one phase cannot crash the whole
    run.  Cleanup code (kill_server, atexit) swallows exceptions to guarantee
    process reaping.  User-visible errors are either propagated or displayed with
    friendly messages.  Each subpackage __init__.py documents its own approach.
"""

__all__: list = []
