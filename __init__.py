"""
llama-server Parameter Optimizer
================================
Multi-phase coordinate descent using Optuna (GP-Bayesian/TPE).

Usage:
  python -m llama_optimizer              # use defaults
  python -m llama_optimizer --model /path/to/model.gguf
  python -m llama_optimizer --server /path/to/llama-server --port 8091
"""
