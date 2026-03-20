"""Pure business-logic layer -- ZERO print/input calls.

Every function takes explicit parameters, returns values, and raises
specific exceptions on failure.  This module sits between the TUI layer
(menu.py, wizard.py, display.py, report.py) and the core
engine/pipeline modules, enabling headless use, testing, and future GUI
or API front-ends.

Implementation is split across focused sub-modules:
- services_config.py  -- config, preset, toggle, and model management
- services_pipeline.py -- pipeline progress, results, database management
- services_command.py  -- command generation and HTML reports

This file re-exports everything for backward compatibility.
"""

# Re-export from services_config
# Re-export state helpers that menu.py accesses via services.get_config / set_config
from ..state import get_config, set_config  # noqa: F401

# Re-export from services_command
from .services_command import (  # noqa: F401
    _FLAG_MAP,
    _MERGE_PHASE_ORDER,
    _build_command_parts,
    _format_command,
    _merge_phase_results,
    generate_html_report,
    generate_optimized_command,
)
from .services_config import (  # noqa: F401
    ConfigValidationError,
    DatabaseResetError,
    ModelInfo,
    ModelSwitchError,
    ServiceError,
    SystemInfo,
    apply_toggle,
    build_arch_config_dense,
    build_arch_config_moe,
    cycle_preset,
    detect_architecture,
    get_available_models,
    get_phase_trial_default,
    get_system_info,
    get_toggle_states,
    save_config_to_disk,
    set_context_size,
    set_draft_model,
    switch_to_model,
)

# Re-export from services_pipeline
from .services_pipeline import (  # noqa: F401
    PhaseDisplayResult,
    PhaseProgress,
    build_phase_base_config,
    delete_study,
    find_resume_point,
    get_model_results,
    get_phase_detail,
    get_phase_results,
    get_pipeline_progress,
    reset_database,
)
