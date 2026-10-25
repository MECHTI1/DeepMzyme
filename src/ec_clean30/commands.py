"""EC trainer commands, capped unit runs and unit verification for the EC class versus CLEAN30 campaign.

``build_unit_command`` turns a concrete recipe and a per-fold view into the exact
``src/train.py`` arguments and environment of one fit. Every data path points into the
view; caches, temporary files and library caches point into the view's ``caches/``
folder; the held-out guard (``DEEPMZYME_FORBIDDEN_READ_ROOTS``) lists the view's
recorded shared and test-bearing roots. The trainer may not generate any input
(``--no-prepare-missing-*``), evaluate a test set, or select a checkpoint by validation
(cosine schedule to zero, terminal checkpoint, campaign training rule). Inner units split
by the view's frozen EC membership and export validation predictions; refit units train
on the whole view with ``--val-fraction 0`` and export predictions of the view's probe
list for the later replay.

Reference argument list of an inner F4 unit (Phase 1 build plan, gate G1.10) followed by
the runner's own additions (``--ring-features-dir``, ``--runs-dir``, ``--run-name``,
``--campaign-run-identity``)::

    --task ec --ec-label-depth 1 --ec-class-vocabulary fixed_seven --ec-conflicting-labels fail
    --site-extraction selected_sites --model-architecture gvp --fusion-mode late_fusion
    --gvp-learning-rate 3e-4 --learning-rate 1e-4 --use-global-esm-branch
    --global-esm-cache-dir V/chain_means_final --global-esm-manifest-sha256 <sha>
    --ec-group-weighting structure_id --ec-class-weight-unit group --batch-size 4 --epochs 1
    --lr-schedule cosine --checkpoint-rule terminal --edge-radius 6.0 --ec-contrastive-weight 0
    --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges --no-prepare-missing-external-features
    --fold-split-source membership --fold-membership-csv V/smoke_membership.csv
    --fold-membership-sha256 <sha> --n-folds 2 --fold-index 0 --export-ec-predictions
    --structure-dir V/structures --summary-csv V/site_summary_train.csv --esm-embeddings-dir V/esm
    --external-features-root-dir V/external --dataset-bundle-id g1-smoke-view --dataset-bundle-sha256 <view sha>
    --device cpu --load-workers 1 --num-workers 0 --seed 42

``check_command`` refuses abbreviated or unknown flags and checks the configuration the
trainer resolves, not only the tokens: every input path inside the view, the runs folder
outside it, no other path, and the same configuration as the recipe's own command.

``run_unit`` executes one command under the workstation cap (``CAP_PREFIX``: 2 cores,
3 GB, no swap) after re-verifying the view and requiring its passed dry run, and writes
``<runs>/<unit>.runner.json`` and ``<runs>/<unit>.log`` next to the trainer's run folder
``<runs>/<unit>/``. ``verify_unit`` checks the receipts a unit must leave (gate G1.10
step 5, plus the prediction files, their structures and the absence of the global branch
where a family has none). ``collect_refit_checkpoints`` re-verifies refit units for a
Freeze-2 manifest, and ``delete_smoke_checkpoints`` removes verified smoke checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

from ec_clean30.campaign import (
    CAMPAIGN_ID, FAMILY_SPECS, KIND_INNER, KIND_REFIT, LEVER_CONFIGS, N_OUTER_FOLDS, STEP_S4, Recipe, canonical_json,
    canonical_sha256, format_learning_rate,
)
from ec_clean30.provenance import (
    SRC_ROOT, HeldOutInputError, is_test_named_path, provenance_record, refuse_test_arguments, sha256_bytes,
    sha256_file, utc_now,
)
from ec_clean30.selection import REFIT_SEEDS
from ec_clean30.views import View, write_receipt_file


TRAIN_SCRIPT = SRC_ROOT / "train.py"
CAP_PREFIX: tuple[str, ...] = ("systemd-run", "--user", "--scope", "-q", "-p", "CPUQuota=200%", "-p", "MemoryMax=3G",
                               "-p", "MemorySwapMax=0")
MEMORY_LIMIT_BYTES = 3 * 1024 ** 3
EC_CLASS_TOKENS = tuple(str(index) for index in range(1, 8))
# Prediction files of the EC export (training.ec_prediction_export.PREDICTION_FILE_NAMES; kept here torch-free).
PREDICTION_FILE_NAMES: Mapping[str, tuple[str, str]] = {
    "validation": ("ec_val_pocket_predictions.csv", "ec_val_protein_predictions.csv"),
    "probe": ("ec_probe_pocket_predictions.csv", "ec_probe_protein_predictions.csv"),
}
GLOBAL_FAMILIES = frozenset(family for family, spec in FAMILY_SPECS.items() if spec.global_branch)

DATA_PATH_FLAGS: tuple[str, ...] = (
    "--structure-dir", "--summary-csv", "--esm-embeddings-dir", "--external-features-root-dir",
    "--ring-features-dir", "--global-esm-cache-dir", "--fold-membership-csv", "--ec-probe-structure-ids",
)
OUTPUT_PATH_FLAGS: tuple[str, ...] = ("--runs-dir",)
CACHE_ENVIRONMENT: Mapping[str, str] = {
    "DEEPMZYME_PARSE_CACHE_DIR": "parse",
    "DEEPMZYME_GRAPH_CACHE_DIR": "graph",
    "TMPDIR": "tmp",
    "MPLCONFIGDIR": "tmp/matplotlib",
    "XDG_CACHE_HOME": "tmp/xdg_cache",
    "TORCH_HOME": "tmp/torch",
}
FIXED_ENVIRONMENT: Mapping[str, str] = {
    "MKL_THREADING_LAYER": "GNU",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "PYTHONHASHSEED": "0",
}
UNSET_ENVIRONMENT: tuple[str, ...] = ("PYTHONPATH", "DEEPMZYME_LOAD_WORKERS", "DEEPMZYME_RUNTIME_CONFIG",
                                      "RING_FEATURES_DIR", "RING_EDGES_DIR", "PYTHONSTARTUP")
REQUIRED_FLAGS: tuple[str, ...] = (
    "--no-prepare-missing-esm-embeddings", "--no-prepare-missing-ring-edges", "--no-prepare-missing-external-features",
    "--export-ec-predictions", "--dataset-bundle-id", "--dataset-bundle-sha256",
)
REFUSED_FLAGS: frozenset[str] = frozenset({
    "--allow-missing-esm-embeddings", "--allow-missing-external-features", "--prepare-missing-ring-edges",
    "--explicit-membership-manifest", "--source-cohort-csv", "--require-all-task-classes", "--use-ring-edges",
    "--require-ring-edges", "--export-validation-predictions", "--save-epoch-checkpoints",
})


class CommandError(ValueError):
    """A unit command cannot be built, or a built command breaks the campaign contract."""


@dataclass(frozen=True)
class UnitCommand:
    """One fit: trainer arguments (after ``src/train.py``), environment overrides and identity."""

    unit_id: str
    kind: str
    step: str
    seed: int
    recipe: Recipe
    argv: tuple[str, ...]
    env: Mapping[str, str]
    identity: Mapping[str, Any]
    unset_env: tuple[str, ...] = UNSET_ENVIRONMENT
    runs_dir: Optional[str] = None
    inner_fold: Optional[int] = None
    device: str = "cpu"

    def command(self, python: str = sys.executable) -> list[str]:
        return [python, str(TRAIN_SCRIPT), *self.argv]

    def environment(self, base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        env = {key: value for key, value in (os.environ if base is None else base).items()
               if key not in self.unset_env}
        env.update(self.env)
        return env

    def to_dict(self) -> dict[str, Any]:
        return {"unit_id": self.unit_id, "kind": self.kind, "step": self.step, "seed": self.seed,
                "inner_fold": self.inner_fold, "device": self.device, "recipe": self.recipe.to_dict(),
                "argv": list(self.argv), "env": dict(self.env), "unset_env": list(self.unset_env),
                "identity": dict(self.identity), "runs_dir": self.runs_dir}


def _value(value: Any) -> str:
    if isinstance(value, bool):
        raise CommandError("Boolean options are bare flags")
    if isinstance(value, float):
        return format_learning_rate(value) if 0 < abs(value) < 1e-2 else repr(value)
    return str(value)


def _render(options: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for flag, value in options.items():
        if value is None or value is False:
            continue
        argv.append(flag)
        if value is not True:
            argv.append(_value(value))
    return argv


def cache_environment(view: View) -> dict[str, str]:
    env = {name: str(view.caches / relative) for name, relative in CACHE_ENVIRONMENT.items()}
    env["DEEPMZYME_FORBIDDEN_READ_ROOTS"] = os.pathsep.join(view.forbidden_read_roots)
    env.update(FIXED_ENVIRONMENT)
    return env


def _global_cache(view: View, recipe: Recipe) -> tuple[Path, str]:
    layer = recipe.chain_mean_layer()
    directory = view.chain_means_dir(layer)
    digest = view.chain_means_manifest_sha256(layer)
    if digest is None:
        raise CommandError(f"{recipe.family} needs the {layer} chain-mean cache; build it in {directory} first "
                           "(src/embed_helpers/esmc_chain_means.py with the view's allowlist).")
    return directory, digest


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


def resolved_config_sha256(argv: Sequence[str]) -> str:
    """Hash of the trainer configuration these arguments resolve to (``training.config.config_to_payload``)."""
    from training.config import config_to_payload, parse_args

    return _payload_sha256(config_to_payload(parse_args(list(argv))))


@lru_cache(maxsize=1)
def trainer_option_strings() -> frozenset[str]:
    """Every exact option string of ``src/train.py`` (abbreviations are not in this set)."""
    from training.config import build_arg_parser

    return frozenset(option for action in build_arg_parser()._actions for option in action.option_strings)


def _base_argv(
    view: View,
    recipe: Recipe,
    *,
    unit_id: str,
    kind: str,
    seed: int,
    runs_dir: str | os.PathLike,
    inner_fold: Optional[int],
    device: str,
) -> tuple[list[str], Optional[str]]:
    """The trainer arguments a recipe gives on ``view`` (without the run identity) and the chain-mean SHA-256."""
    if kind not in (KIND_INNER, KIND_REFIT):
        raise CommandError(f"Unit kind must be {KIND_INNER!r} or {KIND_REFIT!r}, got {kind!r}.")
    spec = recipe.spec
    lr = dict(recipe.learning_rate)
    membership = view.membership
    if kind == KIND_INNER:
        if membership is None:
            raise CommandError("Inner units need an EC membership in the view (smoke membership or inner membership).")
        if inner_fold is None or not 0 <= int(inner_fold) < int(membership["n_folds"]):
            raise CommandError(f"Inner fold {inner_fold!r} is outside 0..{int(membership['n_folds']) - 1}.")
    elif inner_fold is not None:
        raise CommandError("Refit units have no inner fold.")

    head = {"--task": "ec", "--ec-label-depth": 1, "--ec-class-vocabulary": "fixed_seven",
            "--ec-conflicting-labels": "fail", "--site-extraction": "selected_sites",
            "--model-architecture": spec.model_architecture, "--fusion-mode": spec.fusion_mode}
    if "gvp_learning_rate" in lr:
        head["--gvp-learning-rate"] = float(lr["gvp_learning_rate"])
    head["--learning-rate"] = float(lr["learning_rate"])
    cache_digest = None
    if spec.global_branch:
        cache_dir, cache_digest = _global_cache(view, recipe)
        head.update({"--use-global-esm-branch": True, "--global-esm-cache-dir": str(cache_dir),
                     "--global-esm-manifest-sha256": cache_digest})
    model = {"--ec-group-weighting": "structure_id", "--ec-class-weight-unit": "group", "--batch-size": 4,
             "--epochs": int(recipe.epochs), "--lr-schedule": "cosine", "--checkpoint-rule": "terminal",
             "--edge-radius": 6.0, "--ec-contrastive-weight": 0}
    for config_id in recipe.lever_configs:
        for flag, value in LEVER_CONFIGS[config_id].options_for(recipe.family):
            (head if flag in head else model)[flag] = value
    preparation = {"--no-prepare-missing-esm-embeddings": True, "--no-prepare-missing-ring-edges": True,
                   "--no-prepare-missing-external-features": True}
    if kind == KIND_INNER:
        split = {"--fold-split-source": "membership", "--fold-membership-csv": str(view.membership_file()),
                 "--fold-membership-sha256": membership["sha256"], "--n-folds": int(membership["n_folds"]),
                 "--fold-index": int(inner_fold)}
    else:
        split = {"--val-fraction": 0, "--ec-probe-structure-ids": str(view.probe_file),
                 "--ec-probe-structure-ids-sha256": view.probe_sha256}
    tail = {"--export-ec-predictions": True, "--structure-dir": str(view.structures),
            "--summary-csv": str(view.summary), "--esm-embeddings-dir": str(view.esm),
            "--external-features-root-dir": str(view.external), "--dataset-bundle-id": view.bundle_id,
            "--dataset-bundle-sha256": view.manifest_sha256, "--device": device, "--load-workers": 1,
            "--num-workers": 0, "--seed": int(seed)}
    runner = {"--ring-features-dir": str(view.ring), "--runs-dir": str(Path(os.path.abspath(os.fspath(runs_dir)))),
              "--run-name": unit_id}
    argv = [*_render(head), *_render(model), *_render(preparation), *_render(split), *_render(tail),
            *_render(runner)]
    return argv, cache_digest


def build_unit_command(
    view: View,
    recipe: Recipe,
    *,
    unit_id: str,
    kind: str,
    seed: int,
    runs_dir: str | os.PathLike,
    step: str,
    inner_fold: Optional[int] = None,
    device: str = "cpu",
    freeze1_sha256: Optional[str] = None,
    freeze1_amendment_chain: Sequence[str] = (),
    overlay_id: Optional[str] = None,
    overlay_sha256: Optional[str] = None,
    git_head: Optional[str] = None,
    tree_sha256: Optional[str] = None,
) -> UnitCommand:
    """Exact trainer arguments of one unit on ``view`` (see the module docstring for the order).

    The run identity follows map I-10: campaign, Freeze-1 SHA-256 and its amendment chain,
    code tree, overlay ID and SHA-256 (``None`` until the Phase 2 overlay exists), view,
    allowlist, membership or probe list, chain means, fold, inner fold, step, recipe, seed
    and the resolved configuration's SHA-256.
    """
    base_argv, cache_digest = _base_argv(view, recipe, unit_id=unit_id, kind=kind, seed=seed, runs_dir=runs_dir,
                                         inner_fold=inner_fold, device=device)
    membership = view.membership
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "unit_id": unit_id,
        "step": step,
        "kind": kind,
        "fold": view.fold,
        "inner_fold": inner_fold,
        "seed": int(seed),
        "smoke": view.smoke,
        "recipe": recipe.to_dict(),
        "view_manifest_sha256": view.manifest_sha256,
        "allowlist_sha256": view.allowlist_sha256,
        "membership_sha256": membership["sha256"] if (kind == KIND_INNER and membership) else None,
        "probe_sha256": view.probe_sha256 if kind == KIND_REFIT else None,
        "chain_means_manifest_sha256": cache_digest,
        "freeze1_sha256": freeze1_sha256,
        "freeze1_amendment_chain": list(freeze1_amendment_chain),
        "overlay_id": overlay_id,
        "overlay_sha256": overlay_sha256,
        "git_head": git_head,
        "source_tree_sha256": tree_sha256,
        "resolved_config_sha256": resolved_config_sha256(base_argv),
    }
    argv = [*base_argv, "--campaign-run-identity", canonical_json(identity)]
    command = UnitCommand(unit_id=unit_id, kind=kind, step=step, seed=int(seed), recipe=recipe, argv=tuple(argv),
                          env=cache_environment(view), identity=identity,
                          runs_dir=str(Path(os.path.abspath(os.fspath(runs_dir)))), inner_fold=inner_fold,
                          device=device)
    problems = check_command(command, view)
    if problems:
        raise CommandError("Built command breaks the campaign contract:\n  " + "\n  ".join(problems))
    return command


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def _real(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _inside(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    real, base = _real(path), _real(root)
    return real == base or real.startswith(base.rstrip(os.sep) + os.sep)


def _flag_values(argv: Sequence[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for index, token in enumerate(argv):
        if token.startswith("--"):
            flag, _sep, inline = token.partition("=")
            following = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else None
            values.setdefault(flag, []).append(inline if _sep else following)
    return values


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


# Resolved configuration fields that name the fit's inputs; each must lie inside the view.
RESOLVED_DATA_FIELDS: tuple[str, ...] = (
    "structure_dir", "summary_csv", "esm_embeddings_dir", "external_features_root_dir", "ring_features_dir",
    "global_esm_cache_dir", "fold_membership_csv", "ec_probe_structure_ids",
)
ALWAYS_SET_DATA_FIELDS = frozenset({"structure_dir", "summary_csv", "esm_embeddings_dir",
                                    "external_features_root_dir", "ring_features_dir"})


def _expected_settings(command: UnitCommand) -> dict[str, Any]:
    return {
        "task": "ec", "ec_label_depth": 1, "ec_class_vocabulary": "fixed_seven", "ec_conflicting_labels": "fail",
        "site_extraction": "selected_sites", "lr_schedule": "cosine", "checkpoint_rule": "terminal",
        "batch_size": 4, "ec_contrastive_weight": 0.0, "prepare_missing_esm_embeddings": False,
        "prepare_missing_external_features": False, "prepare_missing_ring_edges": False,
        "require_external_features": True, "run_test_eval": False, "test_structure_dir": None,
        "test_summary_csv": None, "export_ec_predictions": True, "use_ring_edges": False,
        "epochs": command.recipe.epochs, "seed": command.seed, "load_workers": 1, "num_workers": 0,
        "run_name": command.unit_id,
    }


def check_command(command: UnitCommand, view: View) -> list[str]:
    """Every contract problem of a unit command (an empty list means it may run).

    Token checks come first (exact trainer option strings only, no test, generation or
    unknown flags, data paths inside the view); then the configuration the trainer resolves
    is checked field by field (every input path inside the view, the runs folder outside it,
    no other path) and compared with the configuration the recipe itself gives on this view.
    """
    from training.config import config_to_payload, parse_args

    problems: list[str] = []
    argv = list(command.argv)
    try:
        refuse_test_arguments(argv)
    except HeldOutInputError as exc:
        problems.append(str(exc))
    options = trainer_option_strings()
    for token in argv:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            if flag not in options:
                problems.append(f"{flag} is not an exact trainer option (abbreviated and unknown flags are refused)")
        elif token.startswith("-") and len(token) > 1 and not _is_number(token):
            problems.append(f"{token} is refused (single-dash options are not used by campaign units)")
    values = _flag_values(argv)
    for flag in sorted(REFUSED_FLAGS & set(values)):
        problems.append(f"refused flag {flag}")
    for flag in REQUIRED_FLAGS:
        if flag not in values:
            problems.append(f"required flag {flag} is missing")
    for flag in DATA_PATH_FLAGS:
        for value in values.get(flag, []):
            if value is None or not _inside(value, view.root):
                problems.append(f"{flag} {value!r} lies outside the view {view.root}")
            elif _inside(value, view.caches) or _inside(value, view.receipts):
                problems.append(f"{flag} {value!r} points into the view's writable caches or receipts")
            elif is_test_named_path(value, relative_to=view.root):
                problems.append(f"{flag} {value!r} is test-named")
    for flag in OUTPUT_PATH_FLAGS:
        for value in values.get(flag, []):
            if value is None or _inside(value, view.root):
                problems.append(f"{flag} {value!r} must be an output folder outside the view")
            elif any(_inside(value, root) for root in view.forbidden_read_roots):
                problems.append(f"{flag} {value!r} lies inside a shared or test-bearing data root")
    if values.get("--dataset-bundle-sha256", [None])[0] != view.manifest_sha256:
        problems.append("--dataset-bundle-sha256 differs from the view manifest SHA-256")
    if values.get("--dataset-bundle-id", [None])[0] != view.bundle_id:
        problems.append("--dataset-bundle-id differs from the view's bundle ID")
    for name, relative in CACHE_ENVIRONMENT.items():
        value = command.env.get(name)
        if value is None or not _inside(value, view.caches):
            problems.append(f"environment {name}={value!r} lies outside the view caches")
    guard = set((command.env.get("DEEPMZYME_FORBIDDEN_READ_ROOTS") or "").split(os.pathsep)) - {""}
    if not set(view.forbidden_read_roots) <= guard:
        problems.append("DEEPMZYME_FORBIDDEN_READ_ROOTS does not cover the view's forbidden roots")
    try:
        config = parse_args(argv)
    except SystemExit as exc:
        problems.append(f"the trainer refuses the arguments (exit {exc.code})")
        return problems
    payload = config_to_payload(config)

    for name, value in _expected_settings(command).items():
        actual = getattr(config, name)
        if actual != value:
            problems.append(f"resolved {name} is {actual!r}, expected {value!r}")
    uses = {"global_esm_cache_dir": command.recipe.spec.global_branch,
            "fold_membership_csv": command.kind == KIND_INNER, "ec_probe_structure_ids": command.kind == KIND_REFIT}
    for name in RESOLVED_DATA_FIELDS:
        value = payload.get(name)
        needed = name in ALWAYS_SET_DATA_FIELDS or uses.get(name, False)
        if value is None:
            if needed:
                problems.append(f"resolved {name} is not set")
            continue
        if not needed:
            problems.append(f"resolved {name} is set, but this unit does not use it")
        if not _inside(value, view.root) or _inside(value, view.caches) or _inside(value, view.receipts):
            problems.append(f"resolved {name} {value!r} lies outside the view's data")
    runs = payload.get("runs_dir")
    if runs is None or _inside(runs, view.root) or any(_inside(runs, root) for root in view.forbidden_read_roots):
        problems.append(f"resolved runs_dir {runs!r} must be an output folder outside the view and the shared roots")
    for name, value in sorted(payload.items()):
        if name in RESOLVED_DATA_FIELDS or name == "runs_dir":
            continue
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(item, str) and os.path.isabs(item):
                problems.append(f"resolved {name} names the path {item!r}; units read only the view's inputs")
    if command.kind == KIND_INNER:
        if config.fold_split_source != "membership" or config.fold_membership_csv is None \
                or _real(config.fold_membership_csv) != _real(view.membership_file()):
            problems.append("inner units must split by the view's EC membership")
    else:
        if config.val_fraction != 0.0 or config.n_folds is not None:
            problems.append("refit units train without validation (--val-fraction 0, no folds)")
        if config.ec_probe_structure_ids is None or _real(config.ec_probe_structure_ids) != _real(view.probe_file):
            problems.append("refit units export the view's probe list")

    # The resolved configuration must be the one the recipe gives on this view.
    actual = dict(payload)
    actual["campaign_run_identity"] = None
    try:
        expected_argv, _digest = _base_argv(
            view, command.recipe, unit_id=command.unit_id, kind=command.kind, seed=command.seed,
            runs_dir=command.runs_dir or payload.get("runs_dir") or "", inner_fold=command.inner_fold,
            device=command.device)
        expected = config_to_payload(parse_args(expected_argv))
    except (CommandError, SystemExit, ValueError) as exc:
        problems.append(f"the recipe's own command cannot be rebuilt on this view: {exc}")
    else:
        differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        if differing:
            problems.append(f"resolved configuration differs from the recipe's command in {differing[:12]}")
    try:
        identity = json.loads(config.campaign_run_identity or "{}")
    except ValueError:
        identity = {}
    if not identity or canonical_json(identity) != canonical_json(dict(command.identity)):
        problems.append("--campaign-run-identity differs from the unit's identity")
    bound = {"campaign_id": CAMPAIGN_ID, "unit_id": command.unit_id, "kind": command.kind, "step": command.step,
             "fold": view.fold, "inner_fold": command.inner_fold, "seed": command.seed, "smoke": view.smoke,
             "recipe": command.recipe.to_dict(), "view_manifest_sha256": view.manifest_sha256,
             "allowlist_sha256": view.allowlist_sha256, "resolved_config_sha256": _payload_sha256(actual)}
    for name, value in bound.items():
        if identity.get(name) != value:
            problems.append(f"the campaign run identity's {name} does not match this unit and view")
    return problems


# ---------------------------------------------------------------------------
# Running one unit
# ---------------------------------------------------------------------------

def runner_receipt_path(runs_dir: Path, unit_id: str) -> Path:
    return Path(runs_dir) / f"{unit_id}.runner.json"


def clear_view_for_unit(view: View, command: UnitCommand, *,
                        provenance: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Re-verify ``view`` and require its passed dry run before ``command`` may run on it.

    The dry-run receipt must name this exact view manifest, record a verified view and the
    same chain means the command reads (a cache rebuilt after the dry run needs a new one).
    Returns the clearance record kept in the runner receipt; raises ``CommandError``.
    """
    from ec_clean30.views import ViewVerificationError, dry_run_problems, read_dry_run_receipt, verify_view

    layer = command.recipe.chain_mean_layer()
    try:
        verification = verify_view(view.root, require_chain_means=(layer,) if layer else (), write_receipt=False,
                                   provenance=provenance or {"note": "verified before a unit run"})
    except ViewVerificationError as exc:
        raise CommandError("Refusing to run: the view does not verify:\n  " + "\n  ".join(exc.problems[:20])) from exc
    receipt = read_dry_run_receipt(view)
    chain_means = {layer: verification["chain_means"][layer]} if layer else None
    problems = dry_run_problems(view, receipt, chain_means=chain_means)
    if problems:
        raise CommandError("Refusing to run: " + "; ".join(problems))
    return {"verified_utc": utc_now(), "view_manifest_sha256": view.manifest_sha256,
            "n_files": verification["n_files"], "chain_means": verification["chain_means"],
            "dry_run_receipt_sha256": receipt["_file_sha256"]}


def run_unit(
    command: UnitCommand,
    *,
    view: View,
    cap_prefix: Sequence[str] = CAP_PREFIX,
    python: str = sys.executable,
    timeout: Optional[float] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run one unit (fresh run folder only) and write its runner receipt and log.

    The command must pass ``check_command``, the view must verify again, and the view must
    hold a passed dry-run receipt for its exact manifest (``clear_view_for_unit``).
    """
    problems = check_command(command, view)
    if problems:
        raise CommandError("Refusing to run:\n  " + "\n  ".join(problems))
    clearance = clear_view_for_unit(view, command, provenance=provenance)
    runs_dir = Path(command.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / command.unit_id
    receipt_path = runner_receipt_path(runs_dir, command.unit_id)
    log_path = runs_dir / f"{command.unit_id}.log"
    for path in (run_dir, receipt_path, log_path):
        if path.exists():
            raise CommandError(f"{path} already exists; a unit runs once into a fresh folder.")
    for relative in CACHE_ENVIRONMENT.values():
        (view.caches / relative).mkdir(parents=True, exist_ok=True)
    full_command = [*cap_prefix, *command.command(python)]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "unit_id": command.unit_id,
        "status": "started",
        "started_utc": utc_now(),
        "command": full_command,
        "cap_prefix": list(cap_prefix),
        "unit": command.to_dict(),
        "view": str(view.root),
        "view_manifest_sha256": view.manifest_sha256,
        "view_clearance": clearance,
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=view.smoke),
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        try:
            completed = subprocess.run(full_command, cwd=str(runs_dir), env=command.environment(), stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    receipt.update({
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "finished_utc": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "children_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024,
        "log": log_path.name,
        "log_sha256": sha256_file(log_path),
    })
    temporary = receipt_path.with_name(receipt_path.name + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt


# ---------------------------------------------------------------------------
# Verifying one unit
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class _Checks:
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: Any = None) -> None:
        self.results[name] = {"ok": bool(ok), "detail": detail}

    @property
    def passed(self) -> bool:
        return all(item["ok"] for item in self.results.values())


def verify_unit(runs_dir: Path, unit_id: str, *, view: Optional[View] = None) -> dict[str, Any]:
    """Check one finished unit against the campaign's receipt contract; returns a per-check record."""
    from training.ec_membership import read_ec_fold_membership

    runs_dir = Path(runs_dir)
    checks = _Checks()
    receipt_path = runner_receipt_path(runs_dir, unit_id)
    if not receipt_path.is_file():
        checks.add("runner_receipt", False, "missing")
        return {"unit_id": unit_id, "passed": False, "checks": checks.results}
    runner = _load_json(receipt_path)
    unit = runner["unit"]
    identity = unit["identity"]
    kind, recipe = unit["kind"], Recipe(unit["recipe"]["family"], tuple(unit["recipe"]["learning_rate"].items()),
                                        int(unit["recipe"]["epochs"]), tuple(unit["recipe"]["lever_configs"]))
    view = view if view is not None else View.load(runner["view"])
    head = {"unit_id": unit_id, "kind": kind, "family": recipe.family, "smoke": identity.get("smoke"),
            "fold": identity.get("fold"), "inner_fold": identity.get("inner_fold"), "seed": identity.get("seed"),
            "step": identity.get("step"), "recipe": recipe.to_dict()}
    checks.add("view_matches", runner["view_manifest_sha256"] == view.manifest_sha256 ==
               identity["view_manifest_sha256"], runner["view_manifest_sha256"])
    checks.add("exit_code", runner.get("returncode") == 0, runner.get("returncode"))
    run_dir = runs_dir / unit_id
    if not run_dir.is_dir():
        checks.add("run_dir", False, "missing")
        return {**head, "passed": False, "checks": checks.results}
    epochs = recipe.epochs
    terminal = run_dir / "terminal_model_checkpoint.pt"
    checks.add("terminal_checkpoint", terminal.is_file() and not (run_dir / "best_model_checkpoint.pt").exists(),
               terminal.name)
    try:
        metadata = _load_json(run_dir / "run_metadata.json")
        run_config = _load_json(run_dir / "run_config.json")
        summary = _load_json(run_dir / "dataset_summary.json")
        diagnostics = _load_json(run_dir / "split_diagnostics.json")
        optimizer = _load_json(run_dir / "optimizer_groups.json")
        prediction = _load_json(run_dir / "ec_prediction_receipt.json")
    except (OSError, ValueError) as exc:
        checks.add("run_files", False, str(exc))
        return {**head, "passed": False, "checks": checks.results}
    checks.add("selected_epoch", metadata.get("selected_checkpoint_epoch") == epochs
               and metadata.get("checkpoint_rule") == "terminal" and metadata.get("fit_status") == "completed",
               metadata.get("selected_checkpoint_epoch"))
    checks.add("history_length", len(run_config.get("history") or []) == epochs, len(run_config.get("history") or []))
    checks.add("no_test_report", metadata.get("test_report") is None and run_config.get("test_report") is None
               and not (run_dir / "test_report.json").exists())
    realised = optimizer.get("realised_total_steps") or []
    planned = optimizer.get("planned_steps_for_epochs_run")
    checks.add("optimizer_groups", optimizer.get("schema_version") == 2 and optimizer.get("realised_steps_match") is True
               and bool(realised) and all(value == planned for value in realised),
               {"planned": planned, "realised": realised})
    runtime = summary.get("runtime_preparation") or {}
    generated = {key: runtime.get(key) for key in ("generated_esm_files", "generated_updated_external_feature_files",
                                                   "generated_ring_edge_files")}
    checks.add("no_generated_features", all(value == 0 for value in generated.values()), generated)
    label_report = diagnostics.get("ec_label_report") or {}
    checks.add("ec_label_report", label_report.get("vocabulary") == "fixed_seven"
               and list(label_report.get("ec_class_tokens") or []) == list(EC_CLASS_TOKENS),
               label_report.get("ec_class_tokens"))
    site_summary = (summary.get("feature_load_report") or {}).get("site_extraction_summary")
    checks.add("site_extraction_summary", bool(site_summary) and site_summary.get("site_extraction") == "selected_sites",
               None if not site_summary else site_summary.get("n_structures"))
    support = diagnostics.get("ec_class_support") or {}
    n_train = sum((support.get("train_proteins_by_ec_class") or {}).values())
    n_val = sum((support.get("val_proteins_by_ec_class") or {}).values())
    loaded = n_train + n_val
    if recipe.family in GLOBAL_FAMILIES:
        global_esm = summary.get("global_esm") or {}
        attach = global_esm.get("attach") or {}
        checks.add("global_attach", attach.get("n_structures") == loaded == global_esm.get("n_structures")
                   and global_esm.get("manifest_sha256") == identity.get("chain_means_manifest_sha256")
                   == (prediction.get("global_esm") or {}).get("manifest_sha256"),
                   {"attached": attach.get("n_structures"), "loaded": loaded})
    else:
        checks.add("no_global_branch", summary.get("global_esm") is None and prediction.get("global_esm") is None
                   and not (run_config.get("config") or {}).get("use_global_esm_branch"),
                   {"dataset_summary": summary.get("global_esm") is not None,
                    "prediction_receipt": prediction.get("global_esm") is not None})
    fold_membership = summary.get("fold_membership") or {}
    mode = "validation" if kind == KIND_INNER else "probe"
    if kind == KIND_INNER:
        membership = view.membership or {}
        checks.add("split_type", fold_membership.get("split_type") == "frozen_ec_membership"
                   and fold_membership.get("membership_file_sha256") == membership.get("sha256")
                   and fold_membership.get("n_train_structures", -1) + fold_membership.get("n_val_structures", -1)
                   == membership.get("n_rows") == loaded,
                   fold_membership.get("split_type"))
        checks.add("ec_predictions", prediction.get("mode") == "validation"
                   and prediction.get("reconciliation_status") == "match", prediction.get("reconciliation_status"))
        try:
            rows = read_ec_fold_membership(view.membership_file(), membership["sha256"]).rows
            inner_fold = int(identity["inner_fold"])
            expected_ids = sorted(row["structure_id"] for row in rows if int(row["fold"]) == inner_fold)
        except (OSError, KeyError, TypeError, ValueError):
            expected_ids = None
    else:
        eligible = view.manifest["eligible"]["n"]
        checks.add("refit_coverage", not fold_membership and n_val == 0 and n_train == eligible
                   and label_report.get("n_structures") == view.manifest["structures"]["n"],
                   {"trained": n_train, "eligible": eligible})
        probe = prediction.get("probe") or {}
        checks.add("ec_predictions", prediction.get("mode") == "probe" and probe.get("file_sha256") == view.probe_sha256
                   and probe.get("n_structures") == view.manifest["probe"]["n"]
                   and prediction.get("reconciliation_status") == "not_applicable",
                   {"mode": prediction.get("mode"), "n_probe": probe.get("n_structures"),
                    "reconciliation_status": prediction.get("reconciliation_status")})
        expected_ids = sorted(view.probe_ids())
    checks.add("prediction_structures", expected_ids is not None
               and prediction.get("structure_ids_sha256") == canonical_sha256(expected_ids),
               None if expected_ids is None else len(expected_ids))
    files: dict[str, Any] = {}
    files_ok = True
    for key, expected_name in zip(("pocket_predictions", "protein_predictions"), PREDICTION_FILE_NAMES[mode]):
        record = prediction.get(key) or {}
        path = run_dir / str(record.get("path") or "")
        ok = record.get("path") == expected_name and path.is_file() and sha256_file(path) == record.get("sha256")
        files[key] = {"path": record.get("path"), "ok": ok}
        files_ok = files_ok and ok
    other_mode = "probe" if mode == "validation" else "validation"
    stray = [name for name in PREDICTION_FILE_NAMES[other_mode] if (run_dir / name).exists()]
    checks.add("prediction_files", files_ok and not stray, {"files": files, "stray": stray})
    checkpoint = prediction.get("checkpoint") or {}
    checks.add("prediction_checkpoint", terminal.is_file() and checkpoint.get("sha256") == sha256_file(terminal)
               and checkpoint.get("epoch") == epochs, checkpoint.get("sha256"))
    profile_path = run_dir / "runtime_profile.json"
    peaks = {"runner_children_peak_rss_bytes": runner.get("children_peak_rss_bytes")}
    if profile_path.is_file():
        peaks["trainer_process_peak_rss_bytes"] = _load_json(profile_path).get("process_peak_rss_bytes")
    recorded = [value for value in peaks.values() if isinstance(value, int)]
    checks.add("peak_memory", all(value < MEMORY_LIMIT_BYTES for value in recorded), peaks)
    checks.add("campaign_identity", metadata.get("campaign_run_identity") == identity, identity.get("unit_id"))
    config = run_config.get("config") or {}
    outside = [key for key in ("structure_dir", "summary_csv", "esm_embeddings_dir", "external_features_root_dir",
                               "ring_features_dir", "global_esm_cache_dir", "fold_membership_csv",
                               "ec_probe_structure_ids")
               if config.get(key) is not None and not _inside(config[key], view.root)]
    checks.add("data_paths_in_view", not outside, outside)
    return {**head, "passed": checks.passed, "checks": checks.results,
            "terminal_checkpoint_sha256": sha256_file(terminal) if terminal.is_file() else None}


VERIFY_UNIT_RECEIPT_NAME = "verify_unit_receipt.json"
SMOKE_CHECKPOINT_DELETION_RECEIPT_NAME = "smoke_checkpoint_deletion_receipt.json"


def verify_units(runs_dir: Path, *, unit_ids: Optional[Sequence[str]] = None,
                 provenance: Optional[Mapping[str, Any]] = None, write: bool = True) -> dict[str, Any]:
    """Verify every unit with a runner receipt in ``runs_dir`` (or the named ones); write one receipt."""
    runs_dir = Path(runs_dir)
    if unit_ids is None:
        unit_ids = sorted(path.name[: -len(".runner.json")] for path in runs_dir.iterdir()
                          if path.name.endswith(".runner.json"))
    if not unit_ids:
        raise CommandError(f"No unit runner receipts in {runs_dir}.")
    units = [verify_unit(runs_dir, unit_id) for unit_id in unit_ids]
    smoke = {unit.get("smoke") for unit in units}
    receipt = {
        "schema_version": 1,
        "status": "passed" if all(unit["passed"] for unit in units) else "failed",
        "runs_dir": str(runs_dir),
        "n_units": len(units),
        "units": units,
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke != {False}),
    }
    if write:
        receipt["receipt_sha256"] = write_receipt_file(runs_dir / VERIFY_UNIT_RECEIPT_NAME, receipt)
    return receipt


def refit_coverage_problems(checkpoints: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why refit records do not make one complete campaign: S4 units, folds 0-4 x seeds 42-46, one recipe per fold."""
    expected = {(fold, seed) for fold in range(N_OUTER_FOLDS) for seed in REFIT_SEEDS}
    pairs = [(item.get("fold"), item.get("seed")) for item in checkpoints]
    problems = []
    if len(pairs) != len(set(pairs)):
        problems.append("a (fold, seed) refit appears more than once")
    missing, extra = sorted(expected - set(pairs)), sorted(set(pairs) - expected, key=str)
    if missing or extra:
        problems.append(f"refits must cover folds 0-{N_OUTER_FOLDS - 1} x seeds {list(REFIT_SEEDS)}; "
                        f"missing {missing[:6]}, unexpected {extra[:6]}")
    steps = sorted({str(item.get("step")) for item in checkpoints} - {STEP_S4})
    if steps:
        problems.append(f"refit units of steps {steps} are not S4 refits")
    for fold in range(N_OUTER_FOLDS):
        recipes = {canonical_json(item.get("recipe")) for item in checkpoints if item.get("fold") == fold}
        if len(recipes) > 1:
            problems.append(f"fold {fold} refits use {len(recipes)} different recipes")
    return problems


def collect_refit_checkpoints(runs_dirs: Sequence[str | os.PathLike], *, smoke: bool) -> list[dict[str, Any]]:
    """Refit checkpoints for a Freeze-2 manifest, each unit verified again now (stored receipts are not trusted).

    A real (non-smoke) manifest also needs complete coverage (``refit_coverage_problems``).
    """
    checkpoints = []
    for runs in runs_dirs:
        runs = Path(runs)
        report = verify_units(runs, provenance={"note": "re-verified for a Freeze-2 manifest"}, write=False)
        for unit in report["units"]:
            if unit.get("kind") != KIND_REFIT:
                continue
            if not unit["passed"]:
                failed = sorted(name for name, item in unit["checks"].items() if not item["ok"])
                raise CommandError(f"{unit['unit_id']} does not pass verify-unit now ({failed}); refused.")
            if bool(unit.get("smoke")) != bool(smoke):
                raise CommandError(f"{unit['unit_id']} is {'a smoke' if unit.get('smoke') else 'a real'} unit; "
                                   f"refused in a {'smoke' if smoke else 'real'} manifest.")
            path = runs / unit["unit_id"] / "terminal_model_checkpoint.pt"
            checkpoints.append({"unit_id": unit["unit_id"], "path": str(path.resolve()),
                                "sha256": unit["terminal_checkpoint_sha256"], "family": unit.get("family"),
                                "fold": unit.get("fold"), "seed": unit.get("seed"), "step": unit.get("step"),
                                "recipe": unit.get("recipe"), "smoke": bool(unit.get("smoke"))})
    if not checkpoints:
        raise CommandError("No refit units in the given runs folders.")
    if not smoke:
        problems = refit_coverage_problems(checkpoints)
        if problems:
            raise CommandError("Refit checkpoints do not make a complete campaign: " + "; ".join(problems))
    return checkpoints


def collect_clean_checkpoints(work_dirs: Sequence[str | os.PathLike], *, smoke: bool) -> list[dict[str, Any]]:
    """Retrained CLEAN checkpoints (arms R1, R2 and C-ESMC) for a Freeze-2 manifest, re-hashed now.

    Each folder is one ``run_clean_split30_comparator.py fit`` work directory. Its final weights
    must still hash to the training receipt of ``fit_manifest.json``, and the receipt must match
    ``prepare_manifest.json``. Inner-fold (gap-arm) fits are refused. A real manifest refuses a fit
    that did not use the published epochs and refresh rate, and needs R1 fits in every outer fold.
    """
    published = _comparator_published_recipe()
    checkpoints, seen = [], set()
    for work_dir in work_dirs:
        work_dir = Path(work_dir)
        try:
            prepare = json.loads((work_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
            fit = json.loads((work_dir / "fit_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CommandError(f"{work_dir} is not a finished CLEAN fit: {exc}") from None
        receipt = fit.get("training_receipt") or {}
        model = str(receipt.get("model_name") or "")
        weights = work_dir / "data" / "model" / f"{model}.pth"
        if not model or not weights.is_file() or sha256_file(weights) != receipt.get("final_weights_sha256"):
            raise CommandError(f"{work_dir}: the final weights do not match the training receipt; refused.")
        if sha256_file(work_dir / "prepare_manifest.json") != fit.get("prepare_manifest_sha256"):
            raise CommandError(f"{work_dir}: prepare_manifest.json changed after the fit; refused.")
        if (int(prepare["seed"]), int(prepare["input_dim"])) != (receipt.get("seed"), receipt.get("input_dim")):
            raise CommandError(f"{work_dir}: the training receipt does not match the prepare manifest; refused.")
        if prepare.get("regime") not in ("r1", "r2") or prepare.get("inner_fold") is not None:
            raise CommandError(f"{model} is an inner-fold (gap-arm) fit; only outer-fold R1, R2 and C-ESMC fits go "
                               "into Freeze 2.")
        recipe = fit.get("recipe") or {}
        if not smoke and {key: recipe.get(key) for key in ("epochs", "adaptive_rate")} != \
                {key: published[key] for key in ("epochs", "adaptive_rate")}:
            raise CommandError(f"{model} was not trained with the published epochs and refresh rate; refused in a "
                               "real manifest.")
        unit_id = f"clean__{model}"
        if unit_id in seen:
            raise CommandError(f"{model} is given twice.")
        seen.add(unit_id)
        checkpoints.append({"unit_id": unit_id, "system": "clean", "path": str(weights.resolve()),
                            "sha256": receipt["final_weights_sha256"], "fold": int(prepare["fold"]),
                            "regime": prepare["regime"], "embedding": prepare["embedding"],
                            "seed": int(prepare["seed"]), "input_dim": int(prepare["input_dim"]),
                            "recipe": recipe, "smoke": bool(smoke)})
    if not smoke:
        missing = sorted(set(range(N_OUTER_FOLDS)) - {item["fold"] for item in checkpoints if item["regime"] == "r1"})
        if missing:
            raise CommandError(f"No CLEAN R1 fit is given for outer folds {missing}.")
    return checkpoints


@lru_cache(maxsize=1)
def _comparator_published_recipe() -> dict[str, Any]:
    """``PUBLISHED_RECIPE`` of the CLEAN comparator script (read by path; the script is not a package)."""
    import importlib.util

    name, path = "run_clean_split30_comparator", SRC_ROOT.parent / "scripts" / "run_clean_split30_comparator.py"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # its dataclasses resolve their module through sys.modules
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    elif Path(module.__file__).resolve() != path.resolve():
        raise CommandError(f"{name} is already loaded from {module.__file__}")
    return dict(module.PUBLISHED_RECIPE)


# Other model checkpoints the trainer may leave next to the terminal one; deleted with it.
SMOKE_EXTRA_CHECKPOINT_NAMES: tuple[str, ...] = ("last_model_checkpoint.pt", "best_model_checkpoint.pt")


def delete_smoke_checkpoints(report: Mapping[str, Any], runs_dir: Path) -> list[dict[str, Any]]:
    """Delete the model checkpoints of verified engineering smoke units (gate G1.10); returns what was deleted.

    Only after every unit passed, only smoke units, and only when the terminal checkpoint still
    has the SHA-256 the verification recorded. The last and best checkpoints, when present, are
    hashed into the record and deleted with it.
    """
    runs_dir = Path(runs_dir)
    if report.get("status") != "passed":
        raise CommandError("Smoke checkpoints are deleted only after every unit passed verify-unit.")
    deleted = []
    for unit in report["units"]:
        if not unit.get("smoke"):
            raise CommandError(f"{unit['unit_id']} is not an engineering smoke unit; its checkpoint is kept.")
        path = runs_dir / unit["unit_id"] / "terminal_model_checkpoint.pt"
        if sha256_file(path) != unit["terminal_checkpoint_sha256"]:
            raise CommandError(f"{path} changed after verification; nothing more is deleted.")
        targets = [(path, unit["terminal_checkpoint_sha256"])]
        targets += [(path.with_name(name), sha256_file(path.with_name(name))) for name in SMOKE_EXTRA_CHECKPOINT_NAMES
                    if path.with_name(name).is_file()]
        for target, digest in targets:
            target.unlink()
            deleted.append({"unit_id": unit["unit_id"], "path": target.relative_to(runs_dir).as_posix(),
                            "sha256": digest})
    return deleted


# ---------------------------------------------------------------------------
# Dry run: every planned command of a view, then preparation probes
# ---------------------------------------------------------------------------

DRY_RUN_PROBE_FAMILIES: tuple[str, ...] = ("F4",)


@dataclass(frozen=True)
class PlannedCommand:
    """A unit command that the dry run checks; ``template_of`` names the plan unit it instantiates."""

    command: Optional[UnitCommand]
    unit_id: str
    kind: str
    family: str
    template_of: Optional[str] = None
    problem: Optional[str] = None


def view_unit_commands(
    view: View,
    *,
    runs_dir: str | os.PathLike,
    exclude_levers: Sequence[str] = (),
    git_head: Optional[str] = None,
    tree_sha256: Optional[str] = None,
) -> list[PlannedCommand]:
    """Commands of every unit planned for ``view``.

    A smoke view plans one 1-epoch inner unit (smoke fold 0) and one refit per family. A
    fold view plans the fold's matrix: S1 units as they are, S2 lever templates for every
    candidate family the lever applies to, S4 refit templates for every candidate family
    (S3 combinations only mix S2 options, which are covered). A command that cannot be built
    is returned with its problem.
    """
    from ec_clean30.campaign import (
        CANDIDATE_FAMILIES, RECIPE_FIXED, RECIPE_S1_WINNER, RECIPE_SELECTED, STEP_S2, STEP_S4, build_fold_plan,
        default_smoke_recipe, plan_units, recipe_for_unit, recipe_with_lever, smoke_unit_id,
    )
    from ec_clean30.selection import FAMILIES, S1_LEARNING_RATE_GRID

    planned: list[PlannedCommand] = []

    def add(recipe: Recipe, *, unit_id: str, kind: str, seed: int, step: str, inner_fold: Optional[int],
            template_of: Optional[str] = None) -> None:
        try:
            command = build_unit_command(view, recipe, unit_id=unit_id, kind=kind, seed=seed, runs_dir=runs_dir,
                                         step=step, inner_fold=inner_fold, git_head=git_head, tree_sha256=tree_sha256)
            planned.append(PlannedCommand(command, unit_id, kind, recipe.family, template_of))
        except (CommandError, ValueError) as exc:
            planned.append(PlannedCommand(None, unit_id, kind, recipe.family, template_of, str(exc)))

    if view.smoke:
        for family in FAMILIES:
            recipe = default_smoke_recipe(family)
            add(recipe, unit_id=smoke_unit_id(view.fold, recipe, refit=False, seed=42), kind=KIND_INNER, seed=42,
                step="smoke", inner_fold=0)
            add(recipe, unit_id=smoke_unit_id(view.fold, recipe, refit=True, seed=42), kind=KIND_REFIT, seed=42,
                step="smoke", inner_fold=None)
        return planned
    excluded = set(exclude_levers)
    for unit in plan_units(build_fold_plan(view.fold)):
        if unit.recipe_source == RECIPE_FIXED:
            add(recipe_for_unit(unit), unit_id=unit.unit_id, kind=unit.kind, seed=unit.seed, step=unit.step,
                inner_fold=unit.inner_fold)
        elif unit.recipe_source == RECIPE_S1_WINNER and unit.step == STEP_S2:
            config_id = unit.lever_configs[0]
            if LEVER_CONFIGS[config_id].lever in excluded:
                continue
            for family in unit.applies_to:
                base = Recipe(family, tuple(S1_LEARNING_RATE_GRID[family][0]))
                add(recipe_with_lever(base, config_id), unit_id=f"{unit.unit_id}__as_{family}", kind=unit.kind,
                    seed=unit.seed, step=unit.step, inner_fold=unit.inner_fold, template_of=unit.unit_id)
        elif unit.recipe_source == RECIPE_SELECTED and unit.step == STEP_S4:
            for family in CANDIDATE_FAMILIES:
                add(Recipe(family, tuple(S1_LEARNING_RATE_GRID[family][0])), unit_id=f"{unit.unit_id}__as_{family}",
                    kind=KIND_REFIT, seed=unit.seed, step=unit.step, inner_fold=None, template_of=unit.unit_id)
    return planned


def evaluate_probe(report: Mapping[str, Any], *, view: View, command: UnitCommand, probe_dir: Path) -> dict[str, Any]:
    """Problems of one preparation probe: reads or writes outside the view, generation, wrong cohort."""
    from ec_clean30.views import classify_probe_records
    from training.ec_membership import read_ec_fold_membership

    classified = classify_probe_records(report, view=view, probe_dir=probe_dir)
    problems: list[str] = []
    if report.get("status") != "prepared" or report.get("returncode") != 0:
        error = report.get("error") or {}
        problems.append(f"preparation did not complete: {error.get('type')}: {str(error.get('message'))[:300]}")
    if report.get("generation_attempts"):
        problems.append(f"feature generation was attempted: {report['generation_attempts']}")
    generated = report.get("generated") or {}
    if any(value not in (0, None) for value in generated.values()):
        problems.append(f"feature files were generated: {generated}")
    for key in ("outside_reads", "outside_listings", "outside_writes", "outside_subprocesses"):
        if classified[key]:
            problems.append(f"{len(classified[key])} {key.replace('_', ' ')}, e.g. {classified[key][:3]}")
    train = set(report.get("train_structure_ids") or [])
    val = set(report.get("val_structure_ids") or [])
    if report.get("status") == "prepared":
        if command.kind == KIND_INNER:
            membership = read_ec_fold_membership(view.membership_file(), view.membership["sha256"])
            fold_index = int(command.identity["inner_fold"])
            expected_val = {row["structure_id"] for row in membership.rows if int(row["fold"]) == fold_index}
            expected_all = {row["structure_id"] for row in membership.rows}
            if train | val != expected_all or val != expected_val:
                problems.append("loaded structures differ from the membership rows")
            if report.get("split_type") != "frozen_ec_membership":
                problems.append(f"split type is {report.get('split_type')!r}, not frozen_ec_membership")
        elif val or train != set(view.eligible_ids()):
            problems.append("refit structures differ from the view's eligible structures")
    return {"unit_id": command.unit_id, "kind": command.kind, "family": command.recipe.family,
            "status": "passed" if not problems else "failed", "problems": problems,
            "returncode": report.get("returncode"), "wall_seconds": report.get("wall_seconds"),
            "n_train_structures": len(train), "n_val_structures": len(val),
            "generation_attempts": report.get("generation_attempts"), "generated": generated,
            "records": {key: value for key, value in classified.items() if not key.startswith("outside")},
            "outside": {key: value[:50] for key, value in classified.items() if key.startswith("outside")},
            "probe_dir": str(probe_dir)}


def dry_run(
    view: View,
    *,
    runs_dir: Optional[str | os.PathLike] = None,
    probe_families: Sequence[str] = DRY_RUN_PROBE_FAMILIES,
    exclude_levers: Sequence[str] = (),
    cap_prefix: Sequence[str] = CAP_PREFIX,
    python: str = sys.executable,
    verify: bool = True,
    timeout: Optional[float] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    write: bool = True,
) -> dict[str, Any]:
    """Check every planned command of ``view`` statically, then probe one inner and one refit unit per family.

    ``verify=False`` skips ``verify_view`` (used only to show that the probe layer catches
    what verification would also catch).
    """
    from ec_clean30.views import DRY_RUN_RECEIPT_NAME, run_probe, verify_view

    runs_dir = Path(os.path.abspath(os.fspath(runs_dir))) if runs_dir is not None else view.root.parent / "runs"
    verification = None
    if verify:
        verification = verify_view(view.root, require_chain_means=("final",), write_receipt=True,
                                   provenance=provenance)
    planned = view_unit_commands(view, runs_dir=runs_dir, exclude_levers=exclude_levers)
    static_failures = [{"unit_id": item.unit_id, "problem": item.problem} for item in planned if item.problem]
    probes = []
    stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
    probe_root = view.caches / "dry_run" / f"run_{stamp}_{os.getpid()}"
    for family in probe_families:
        for kind in (KIND_INNER, KIND_REFIT):
            candidates = [item for item in planned if item.command is not None and item.family == family
                          and item.kind == kind and not item.command.recipe.lever_configs]
            if not candidates:
                probes.append({"family": family, "kind": kind, "status": "failed",
                               "problems": [f"no buildable {kind} command of {family} to probe"]})
                continue
            command = candidates[0].command
            report = run_probe(view=view, unit_id=command.unit_id, argv=command.argv, env=command.env,
                               probe_root=probe_root, cap_prefix=cap_prefix, python=python, timeout=timeout)
            probes.append(evaluate_probe(report, view=view, command=command, probe_dir=probe_root / command.unit_id))
    passed = not static_failures and all(probe["status"] == "passed" for probe in probes)
    receipt = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "view": str(view.root),
        "view_manifest_sha256": view.manifest_sha256,
        "kind": view.kind,
        "fold": view.fold,
        "verified_view": verification is not None,
        "verification": None if verification is None else {
            key: verification.get(key) for key in ("view_manifest_sha256", "n_files", "n_structures", "n_eligible",
                                                   "membership", "chain_means")},
        "n_commands_checked": len(planned),
        "n_template_commands": sum(1 for item in planned if item.template_of),
        "static_failures": static_failures,
        "excluded_levers": list(exclude_levers),
        "runs_dir_checked": str(runs_dir),
        "probes": probes,
        "cap_prefix": list(cap_prefix),
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=view.smoke),
    }
    if write:
        receipt["receipt_sha256"] = write_receipt_file(view.receipts / DRY_RUN_RECEIPT_NAME, receipt)
    return receipt
