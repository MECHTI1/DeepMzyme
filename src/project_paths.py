from __future__ import annotations

import os
from pathlib import Path

MEDIA_DATA_ROOT_ENV = "DEEPGM_MEDIA_DATA_ROOT"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_NAME = "DeepMzyme_Data"
COLAB_BUNDLES_DIR_NAME = "DeepMzyme_Colab_Bundles"
STRUCTURE_STORE_DIR_NAME = "structure_store"

DATA_DIR = PROJECT_ROOT / DATA_DIR_NAME
COLAB_BUNDLES_DIR = DATA_DIR / COLAB_BUNDLES_DIR_NAME
STRUCTURE_STORE_DIR = DATA_DIR / STRUCTURE_STORE_DIR_NAME
ESM_EMBEDDINGS_DIR = DATA_DIR / "esm_embeddings"
EMBEDDINGS_DIR = ESM_EMBEDDINGS_DIR
RING_FEATURES_DIR = DATA_DIR / "RING_features"
UPDATED_FEATURE_EXTRACTION_DIR = DATA_DIR / "updated_feature_extraction"
RUNS_DIR = DATA_DIR / "training_runs"
MEDIA_DATA_ROOT = Path(os.getenv(MEDIA_DATA_ROOT_ENV, "/media/Data")).expanduser()
PINMYMETAL_SETS_DIR = MEDIA_DATA_ROOT / "pinmymetal_sets"
MAHOMES_DIR = PINMYMETAL_SETS_DIR / "mahomes"
MAHOMES_TRAIN_SET_DIR = MAHOMES_DIR / "train_set"
MAHOMES_SUMMARY_DIR = MAHOMES_TRAIN_SET_DIR / "data_summarizing_tables"

SUMMARY_TABLE_CSV = MAHOMES_SUMMARY_DIR / "data_summarazing_table.csv"
TRANSITION_METALS_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "data_summarazing_table_transition_metals.csv"
PREDICTION_RESULTS_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "prediction_results_summary.csv"
WHETHER_CATALYTIC_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "data_summarazing_table_transition_metals_whether_catalytic.csv"
CATALYTIC_ONLY_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "final_data_summarazing_table_transition_metals_only_catalytic.csv"


def get_default_embeddings_dir() -> Path:
    return EMBEDDINGS_DIR


def get_default_ring_features_dir() -> Path:
    return RING_FEATURES_DIR


def get_default_updated_feature_extraction_dir() -> Path:
    return UPDATED_FEATURE_EXTRACTION_DIR


def get_default_runs_dir() -> Path:
    return RUNS_DIR


def get_default_structure_store_dir() -> Path:
    return STRUCTURE_STORE_DIR


def _resolve_project_dir(
    configured_dir: str | None,
    default_dir: Path,
    create: bool = True,
) -> Path:
    if configured_dir:
        resolved_dir = Path(configured_dir).expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = PROJECT_ROOT / resolved_dir
    else:
        resolved_dir = default_dir

    if create:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir


def resolve_embeddings_dir(configured_dir: str | None, create: bool = True) -> Path:
    return _resolve_project_dir(
        configured_dir=configured_dir,
        default_dir=get_default_embeddings_dir(),
        create=create,
    )


def resolve_ring_features_dir(configured_dir: str | None, create: bool = True) -> Path:
    return _resolve_project_dir(
        configured_dir=configured_dir,
        default_dir=get_default_ring_features_dir(),
        create=create,
    )


def resolve_runs_dir(configured_dir: str | None, create: bool = True) -> Path:
    return _resolve_project_dir(
        configured_dir=configured_dir,
        default_dir=get_default_runs_dir(),
        create=create,
    )


RELOCATED_FILES: dict[str, str] = {
    # Compute profiling utilities (bench/)
    "benchmark_step.py": "bench/benchmark_step.py",
    "benchmark_step_realistic.py": "bench/benchmark_step_realistic.py",
    "build_realistic_benchmark_subset.py": "bench/build_realistic_benchmark_subset.py",
    "benchmark_pmm_gpu_concurrency.py": "bench/benchmark_pmm_gpu_concurrency.py",
    # Read-only diagnostics and saved-result analysis (tools/analysis/)
    "audit_pmm_replay.py": "tools/analysis/analyze_pmm_replay.py",
    "audit_pmm_screen_agreement.py": "tools/analysis/analyze_pmm_screen_agreement.py",
    "summarize_pmm_core_diagnostic.py": "tools/analysis/summarize_pmm_core_diagnostic.py",
    "audit_v3_early_stopping.py": "tools/analysis/analyze_early_stopping.py",
    "audit_v3_effective_decay.py": "tools/analysis/analyze_effective_weight_decay.py",
    "audit_v3_epoch_contrast.py": "tools/analysis/analyze_epoch_contrast.py",
    "audit_v3_prefit.py": "tools/analysis/analyze_prefit_diagnostics.py",
    "audit_v3_regression.py": "tools/analysis/analyze_regression_parity.py",
    "audit_v3_step_d_readiness.py": "tools/analysis/analyze_regularization_readiness.py",
    "audit_v3_train_val_gap.py": "tools/analysis/analyze_train_validation_gap.py",
    # Dataset verification and fold construction (scripts/data/)
    "scripts/build_v3_folds.py": "scripts/data/build_pinmymetal_folds.py",
    # Thin benchmark CLI entry points (scripts/benchmarks/)
    "run_pmm_core_campaign.py": "scripts/benchmarks/run_pinmymetal_context_campaign.py",
    "run_pmm_five_class_screen.py": "scripts/benchmarks/run_pinmymetal_five_class_screen.py",
    "run_pmm_v3_campaign.py": "scripts/benchmarks/run_pinmymetal_comparison.py",
    "run_pmm_v3_sched10.py": "scripts/benchmarks/run_pinmymetal_schedule_comparison.py",
    "run_pmm_v3_step_f.py": "scripts/benchmarks/run_pinmymetal_reference_eval.py",
    "run_pmm_trunklr.py": "scripts/benchmarks/run_pinmymetal_encoder_lr.py",
    "run_metal_completion.py": "scripts/benchmarks/run_metal_model_comparison.py",
    # Context-controlled comparison protocol (src/benchmarking/pinmymetal/source_context/)
    "pmm_core_assessment.py": "src/benchmarking/pinmymetal/source_context/assessment.py",
    "pmm_core_replay.py": "src/benchmarking/pinmymetal/source_context/replay.py",
    # Near-copy-grouped comparison protocol (src/benchmarking/pinmymetal/near_copy/)
    "src/benchmarking/v3_folds.py": "src/benchmarking/pinmymetal/near_copy/folds.py",
    "pmm_v3_campaign.py": "src/benchmarking/pinmymetal/near_copy/campaign.py",
    "pmm_v3_assessment.py": "src/benchmarking/pinmymetal/near_copy/assessment.py",
    "pmm_v3_bundle.py": "src/benchmarking/pinmymetal/near_copy/bundle.py",
    "pmm_v3_probes.py": "src/benchmarking/pinmymetal/near_copy/probes.py",
    "pmm_v3_speed_report.py": "src/benchmarking/pinmymetal/near_copy/speed_report.py",
    "pmm_v3_host_pull.py": "src/benchmarking/pinmymetal/near_copy/host_pull.py",
    "pmm_v3_step_b.py": "src/benchmarking/pinmymetal/near_copy/throughput_stage.py",
    "pmm_v3_step_c.py": "src/benchmarking/pinmymetal/near_copy/screening_stage.py",
    "pmm_v3_step_d.py": "src/benchmarking/pinmymetal/near_copy/regularization_stage.py",
    "pmm_v3_step_e2.py": "src/benchmarking/pinmymetal/near_copy/confirmation_stage.py",
    "pmm_v3_e2_assessment.py": "src/benchmarking/pinmymetal/near_copy/confirmation_assessment.py",
    "pmm_v3_sched10.py": "src/benchmarking/pinmymetal/near_copy/schedule_comparison.py",
    "pmm_v3_sched10_assessment.py": "src/benchmarking/pinmymetal/near_copy/schedule_assessment.py",
    "pmm_v3_test_prep.py": "src/benchmarking/pinmymetal/near_copy/test_preparation.py",
    "pmm_v3_step_f.py": "src/benchmarking/pinmymetal/near_copy/reference_evaluation.py",
    "pmm_v3_step_f_report.py": "src/benchmarking/pinmymetal/near_copy/reference_report.py",
    # Encoder learning-rate comparison (src/benchmarking/pinmymetal/encoder_learning_rate/)
    "pmm_trunklr.py": "src/benchmarking/pinmymetal/encoder_learning_rate/campaign.py",
    "pmm_trunklr_assessment.py": "src/benchmarking/pinmymetal/encoder_learning_rate/assessment.py",
    "pmm_trunklr_v2.py": "src/benchmarking/pinmymetal/encoder_learning_rate/followup_campaign.py",
    "pmm_trunklr_assessment_v2.py": "src/benchmarking/pinmymetal/encoder_learning_rate/followup_assessment.py",
    "pmm_trunklr_safeguards_v2.py": "src/benchmarking/pinmymetal/encoder_learning_rate/safeguards.py",
    # Subsequent controlled model comparisons (src/benchmarking/pinmymetal/model_comparison/)
    "metal_completion.py": "src/benchmarking/pinmymetal/model_comparison/coordinator.py",
    "metal_completion_campaign.py": "src/benchmarking/pinmymetal/model_comparison/campaign.py",
    "metal_completion_assessment.py": "src/benchmarking/pinmymetal/model_comparison/assessment.py",
    "metal_completion_evaluation.py": "src/benchmarking/pinmymetal/model_comparison/evaluation.py",
    "metal_completion_safety.py": "src/benchmarking/pinmymetal/model_comparison/safety.py",
    "metal_completion_session.py": "src/benchmarking/pinmymetal/model_comparison/session.py",
    # Relocated benchmark tests (tests/benchmarking/)
    "tests/test_v3_assessment.py": "tests/benchmarking/test_pinmymetal_near_copy_assessment.py",
    "tests/test_v3_audit.py": "tests/benchmarking/test_pinmymetal_regression_parity.py",
    "tests/test_v3_bundle.py": "tests/benchmarking/test_pinmymetal_near_copy_bundle.py",
    "tests/test_v3_campaign.py": "tests/benchmarking/test_pinmymetal_near_copy_campaign.py",
    "tests/test_v3_decay_audit.py": "tests/benchmarking/test_pinmymetal_effective_weight_decay.py",
    "tests/test_v3_early_stopping_audit.py": "tests/benchmarking/test_pinmymetal_early_stopping.py",
    "tests/test_v3_epoch_contrast.py": "tests/benchmarking/test_pinmymetal_epoch_contrast.py",
    "tests/test_v3_extension.py": "tests/benchmarking/test_pinmymetal_near_copy_extension.py",
    "tests/test_v3_folds.py": "tests/benchmarking/test_pinmymetal_near_copy_folds.py",
    "tests/test_v3_model_options.py": "tests/benchmarking/test_pinmymetal_model_options.py",
    "tests/test_v3_replay_policy.py": "tests/benchmarking/test_pinmymetal_replay_policy.py",
    "tests/test_v3_sched10.py": "tests/benchmarking/test_pinmymetal_schedule_comparison.py",
    "tests/test_v3_speed_report.py": "tests/benchmarking/test_pinmymetal_speed_report.py",
    "tests/test_v3_step_b_tools.py": "tests/benchmarking/test_pinmymetal_throughput_stage.py",
    "tests/test_v3_step_c_launcher.py": "tests/benchmarking/test_pinmymetal_screening_stage.py",
    "tests/test_v3_step_d_launcher.py": "tests/benchmarking/test_pinmymetal_regularization_stage.py",
    "tests/test_v3_step_e2.py": "tests/benchmarking/test_pinmymetal_confirmation_stage.py",
    "tests/test_v3_step_f.py": "tests/benchmarking/test_pinmymetal_reference_evaluation.py",
    "tests/test_v3_training_options.py": "tests/benchmarking/test_pinmymetal_training_options.py",
    "tests/test_v3_train_val_gap_audit.py": "tests/benchmarking/test_pinmymetal_train_validation_gap.py",
    "tests/test_trunklr.py": "tests/benchmarking/test_pinmymetal_encoder_lr.py",
    "tests/test_trunklr_safeguards_v2.py": "tests/benchmarking/test_pinmymetal_encoder_lr_safeguards.py",
}


def resolve_repo_file(root: Path | str, name: str | Path) -> Path:
    root_path = Path(root)
    rel = str(name).replace("\\", "/")
    direct = root_path / rel
    if direct.exists():
        return direct
    mapped = RELOCATED_FILES.get(rel)
    if mapped is not None:
        candidate = root_path / mapped
        if candidate.exists():
            return candidate
    return direct


def resolve_existing_path(path: Path | str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    for depth in (1, 2, 3):
        if len(parts) > depth:
            rel = "/".join(parts[-depth:])
            mapped = RELOCATED_FILES.get(rel)
            if mapped is not None:
                candidate = p.parents[depth - 1] / mapped
                if candidate.exists():
                    return candidate
    return p


def _build_module_aliases() -> dict[str, Path]:
    aliases: dict[str, Path] = {}
    for old_rel, new_rel in RELOCATED_FILES.items():
        target = PROJECT_ROOT / new_rel
        stem = Path(old_rel).stem
        if "/" not in old_rel:
            aliases[stem] = target
        elif old_rel == "src/benchmarking/v3_folds.py":
            aliases["benchmarking.v3_folds"] = target
            aliases["src.benchmarking.v3_folds"] = target
        elif old_rel == "scripts/build_v3_folds.py":
            aliases["build_v3_folds"] = target
            aliases["scripts.build_v3_folds"] = target
        elif old_rel.startswith("tests/"):
            aliases[stem] = target
            aliases[f"tests.{stem}"] = target
    return aliases


_MODULE_ALIASES = _build_module_aliases()


class _RelocatedModuleFinder:
    """MetaPathFinder resolving historical flat module names to their organized package locations."""

    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None):
        import importlib.util

        target_path = _MODULE_ALIASES.get(fullname)
        if target_path is not None and target_path.is_file():
            return importlib.util.spec_from_file_location(fullname, str(target_path))
        return None


_ALIASES_INSTALLED = False


def install_relocated_module_aliases() -> None:
    global _ALIASES_INSTALLED
    import importlib.util
    import sys

    for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if not any(getattr(f, "__name__", "") == "_RelocatedModuleFinder" or isinstance(f, _RelocatedModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _RelocatedModuleFinder)
    if not _ALIASES_INSTALLED:
        orig_spec = importlib.util.spec_from_file_location

        def _spec_from_file_location(name, location=None, *args, **kwargs):
            if location is not None:
                resolved = resolve_existing_path(location)
                location = str(resolved) if isinstance(location, str) else resolved
            return orig_spec(name, location, *args, **kwargs)

        importlib.util.spec_from_file_location = _spec_from_file_location
        _ALIASES_INSTALLED = True


install_relocated_module_aliases()

