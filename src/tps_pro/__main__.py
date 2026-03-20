"""Allow running as: python -m tps_pro"""

from __future__ import annotations

from .main import main


def _suppress_optuna_warnings() -> None:
    import warnings

    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


if __name__ == "__main__":
    _suppress_optuna_warnings()
    main()
