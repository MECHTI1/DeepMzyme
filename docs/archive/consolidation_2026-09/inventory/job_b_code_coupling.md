# Job B code/document consumers

Base: `3c0f80c`; command per tracked Markdown basename: `rg --no-ignore -n -F BASENAME src scripts tests notebooks ROOT_PY_FILES .agents CLEAN prepare_training_and_test_set`. Generic README matches deliberately over-include consumers. No new path dependency beyond PLAN_v2 §4.2 was found in edited owners; files keep their existing paths.

## .agents/skills/gpu-use-skill/references/monitor.md

```text
.agents/skills/gpu-use-skill/SKILL.md:199:it. Read [the monitor handoff](references/monitor.md) only when delegating.
```

## AGENTS.md

```text
.agents/skills/gpu-use-skill/SKILL.md:15:1. Read the latest user request, root `AGENTS.md`,
.agents/skills/gpu-use-skill/SKILL.md:41:  `~/deepmzyme-vm/AGENTS.md` and current `config.env`. All lifecycle actions go
.agents/skills/gpu-use-skill/SKILL.md:179:  If accepted work exceeds its ceiling, follow root `AGENTS.md`'s budget rule;
```

## EXPERIMENT_STATUS.md

```text
.agents/skills/gpu-use-skill/SKILL.md:16:   [current status](../../../EXPERIMENT_STATUS.md), and the applicable campaign
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
notebooks/DeepMzyme_training_colab.ipynb:12:    "> **Live values are not canonical stage defaults.** The assignments in **Main configuration** are the current editable/resume state. For a numbered metal stage, paste exactly one block from [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md). Read [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) before choosing a stage; use the [configuration guide](../docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) for option semantics and [`docs/DATASETS.md`](../docs/DATASETS.md) for split/test provenance.\n",
```

## Plan.md

```text
scripts/prepare_colab_smoke_snapshot.py:28:         "src", "requirements", "tests", "scripts", "notebooks", "docs", "Plan.md",
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:326:        "split defined in `Plan.md`.",
notebooks/DeepMzyme_training_colab.ipynb:1459:    "    return (path / \"src\" / \"train.py\").is_file() and (path / \"Plan.md\").is_file()\n",
CLEAN/train_clean_predictor_baselines.ipynb:50:    "        if (path / \"Plan.md\").exists() and (path / \"DeepMzyme_Data\").exists():\n",
CLEAN/train_clean_predictor_baselines.ipynb:53:    "        if (path / \"Plan.md\").exists():\n",
tests/test_serial_metal_evidence.py:61:        "docs/Plan.md": "docs"}}
prepare_training_and_test_set/provenance/common_pdbid_70_30/generated_README.md:8:split defined in `Plan.md`.
```

## README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
```

## docs/COLAB_GPU_RUNBOOK.md

```text
.agents/skills/gpu-use-skill/SKILL.md:44:- Colab: [Colab runbook](../../../docs/COLAB_GPU_RUNBOOK.md) and the available
```

## docs/DATASETS.md

```text
notebooks/DeepMzyme_training_colab.ipynb:12:    "> **Live values are not canonical stage defaults.** The assignments in **Main configuration** are the current editable/resume state. For a numbered metal stage, paste exactly one block from [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md). Read [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) before choosing a stage; use the [configuration guide](../docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) for option semantics and [`docs/DATASETS.md`](../docs/DATASETS.md) for split/test provenance.\n",
notebooks/DeepMzyme_training_colab.ipynb:155:    "# v12 includes complete audited CARE ESM/external/RING caches; see docs/DATASETS.md.\n",
notebooks/DeepMzyme_training_colab.ipynb:2077:    "        \"The default DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz bundle contains exact, historical non-overlapped, and Common-PDBID 70/30 PinMyMetal, CLEAN_30 folds 0-4 with the conservative CLEAN_30_main source, the original CLEAN_30_shared source, and CARE Task 1 clusterRes30 (incomplete feature caches; see docs/DATASETS.md); \"\n",
prepare_training_and_test_set/provenance/README.md:13:recorded in `docs/DATASETS.md`.
```

## docs/EC_TRAINING_PIPELINE_PLAYBOOK.md

```text
src/run_ec_baselines.py:84:    playbook = root / "docs/EC_TRAINING_PIPELINE_PLAYBOOK.md"
```

## docs/GCP_GPU_RUNBOOK.md

```text
.agents/skills/gpu-use-skill/SKILL.md:40:- GCP: [GCP runbook](../../../docs/GCP_GPU_RUNBOOK.md), then the installed
.agents/skills/gpu-use-skill/SKILL.md:77:[GCP recovery procedure](../../../docs/GCP_GPU_RUNBOOK.md#recover-a-stopped-vm-in-the-same-region)
```

## docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md

```text
notebooks/DeepMzyme_training_colab.ipynb:12:    "> **Live values are not canonical stage defaults.** The assignments in **Main configuration** are the current editable/resume state. For a numbered metal stage, paste exactly one block from [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md). Read [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) before choosing a stage; use the [configuration guide](../docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) for option semantics and [`docs/DATASETS.md`](../docs/DATASETS.md) for split/test provenance.\n",
notebooks/DeepMzyme_training_colab.ipynb:5040:    "Detailed artifact schemas and troubleshooting guidance are maintained in the [configuration guide](../docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md)."
```

## docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md

```text
src/serial_metal_campaign/profile.py:195:        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
src/run_metal_ring_pilot.py:117:                                                                  root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
src/run_metal_coordination_geometry_pilot.py:93:        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
src/run_metal_architecture_pilot.py:151:    playbook = (root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md").read_text()
src/run_metal_architecture_pilot.py:240:        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
tests/test_standalone_baselines.py:196:    name = "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"
notebooks/DeepMzyme_training_colab.ipynb:12:    "> **Live values are not canonical stage defaults.** The assignments in **Main configuration** are the current editable/resume state. For a numbered metal stage, paste exactly one block from [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md). Read [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) before choosing a stage; use the [configuration guide](../docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) for option semantics and [`docs/DATASETS.md`](../docs/DATASETS.md) for split/test provenance.\n",
notebooks/DeepMzyme_training_colab.ipynb:541:    "    playbook = (snapshot_root / \"docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md\").read_text()\n",
notebooks/DeepMzyme_training_colab.ipynb:9243:    "Use the exact Stage 6 control block, bootstrap settings, and decision gate from the [metal playbook](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md)."
notebooks/DeepMzyme_training_colab.ipynb:11797:    "Stage 7 writes a new output folder and must not modify Stage 6/6B artifacts or feed test metrics back into model, checkpoint, architecture, calibration, threshold, or hyperparameter choices. See the [Stage 7 playbook block](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md) for the one-shot checklist."
```

## docs/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md

```text
scripts/colab_artifact_streamer.py:8:``docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md`` section 7).
```

## docs/archive/consolidation_2026-09/baseline/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/archive/consolidation_2026-09/verification/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/archive/workflows/list_train_commands_legacy.md

```text
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
```

## docs/notebook_outputs/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/colab_care_cache_smoke_20260914/care_smoke_stock/late_fusion/runs/standalone_ec1_care30_late_fusion_smoke_v1/active_run_config.md

```text
src/stage6_standalone.py:961:    (run_dir / "active_run_config.md").write_text(
notebooks/DeepMzyme_training_colab.ipynb:3787:    "    md_path = target_dir / \"active_run_config.md\"\n",
src/training/run.py:164:        "active_run_config.md",
tests/smoke_checks.py:246:        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
```

## docs/notebook_outputs/raw/colab_care_cache_smoke_20260914/care_smoke_stock/only_esm/runs/standalone_ec1_care30_only_esm_smoke_v1/active_run_config.md

```text
notebooks/DeepMzyme_training_colab.ipynb:3787:    "    md_path = target_dir / \"active_run_config.md\"\n",
tests/smoke_checks.py:246:        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
src/training/run.py:164:        "active_run_config.md",
src/stage6_standalone.py:961:    (run_dir / "active_run_config.md").write_text(
```

## docs/notebook_outputs/raw/colab_care_cache_smoke_20260914/care_smoke_stock/only_gvp/runs/standalone_ec1_care30_only_gvp_smoke_v1/active_run_config.md

```text
notebooks/DeepMzyme_training_colab.ipynb:3787:    "    md_path = target_dir / \"active_run_config.md\"\n",
tests/smoke_checks.py:246:        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
src/training/run.py:164:        "active_run_config.md",
src/stage6_standalone.py:961:    (run_dir / "active_run_config.md").write_text(
```

## docs/notebook_outputs/raw/ec1_standalone_v12_20260914/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/ec1_standalone_v12_20260914/runs/standalone_ec1_care30_late_fusion_baseline_v1/active_run_config.md

```text
src/stage6_standalone.py:961:    (run_dir / "active_run_config.md").write_text(
notebooks/DeepMzyme_training_colab.ipynb:3787:    "    md_path = target_dir / \"active_run_config.md\"\n",
src/training/run.py:164:        "active_run_config.md",
tests/smoke_checks.py:246:        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
```

## docs/notebook_outputs/raw/ec1_standalone_v12_20260914/runs/standalone_ec1_care30_only_esm_baseline_v1/active_run_config.md

```text
src/stage6_standalone.py:961:    (run_dir / "active_run_config.md").write_text(
notebooks/DeepMzyme_training_colab.ipynb:3787:    "    md_path = target_dir / \"active_run_config.md\"\n",
src/training/run.py:164:        "active_run_config.md",
tests/smoke_checks.py:246:        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
```

## docs/notebook_outputs/raw/ec1_standalone_v12_20260914/runs/standalone_ec1_care30_only_gvp_baseline_v1/active_run_config.md

```text
src/stage6_standalone.py:961:    (run_dir / "active_run_config.md").write_text(
notebooks/DeepMzyme_training_colab.ipynb:3787:    "    md_path = target_dir / \"active_run_config.md\"\n",
tests/smoke_checks.py:246:        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
src/training/run.py:164:        "active_run_config.md",
```

## docs/notebook_outputs/raw/legacy_nonoverlap_test_access/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
```

## docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/finalization/decision_record.md

```text
src/run_metal_ring_pilot.py:356:    (output / "ring_decision_record.md").write_text(
src/run_metal_coordination_geometry_pilot.py:409:    (output / "geometry_decision_record.md").write_text(
src/run_metal_architecture_pilot.py:794:    (output / "decision_record.md").write_text(text)
```

## docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_021/decision_record.md

```text
src/run_metal_coordination_geometry_pilot.py:409:    (output / "geometry_decision_record.md").write_text(
src/run_metal_architecture_pilot.py:794:    (output / "decision_record.md").write_text(text)
src/run_metal_ring_pilot.py:356:    (output / "ring_decision_record.md").write_text(
```

## docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_029/decision_record.md

```text
src/run_metal_ring_pilot.py:356:    (output / "ring_decision_record.md").write_text(
src/run_metal_coordination_geometry_pilot.py:409:    (output / "geometry_decision_record.md").write_text(
src/run_metal_architecture_pilot.py:794:    (output / "decision_record.md").write_text(text)
```

## docs/notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
```

## docs/notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/finalization/geometry_decision_record.md

```text
src/run_metal_coordination_geometry_pilot.py:409:    (output / "geometry_decision_record.md").write_text(
```

## docs/notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/geometry_decision_record.md

```text
src/run_metal_coordination_geometry_pilot.py:409:    (output / "geometry_decision_record.md").write_text(
```

## docs/notebook_outputs/raw/metal_ec1_association_20260915/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/metal_ring_pilot_20260915/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/metal_ring_pilot_20260915/finalization/ring_decision_record.md

```text
src/run_metal_ring_pilot.py:356:    (output / "ring_decision_record.md").write_text(
```

## docs/notebook_outputs/raw/metal_ring_pilot_20260915/finalization/source/docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md

```text
src/run_metal_ring_pilot.py:117:                                                                  root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
src/serial_metal_campaign/profile.py:195:        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
notebooks/DeepMzyme_training_colab.ipynb:12:    "> **Live values are not canonical stage defaults.** The assignments in **Main configuration** are the current editable/resume state. For a numbered metal stage, paste exactly one block from [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md). Read [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) before choosing a stage; use the [configuration guide](../docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) for option semantics and [`docs/DATASETS.md`](../docs/DATASETS.md) for split/test provenance.\n",
notebooks/DeepMzyme_training_colab.ipynb:541:    "    playbook = (snapshot_root / \"docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md\").read_text()\n",
notebooks/DeepMzyme_training_colab.ipynb:9243:    "Use the exact Stage 6 control block, bootstrap settings, and decision gate from the [metal playbook](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md)."
notebooks/DeepMzyme_training_colab.ipynb:11797:    "Stage 7 writes a new output folder and must not modify Stage 6/6B artifacts or feed test metrics back into model, checkpoint, architecture, calibration, threshold, or hyperparameter choices. See the [Stage 7 playbook block](../docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md) for the one-shot checklist."
src/run_metal_architecture_pilot.py:151:    playbook = (root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md").read_text()
src/run_metal_architecture_pilot.py:240:        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
tests/test_standalone_baselines.py:196:    name = "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"
src/run_metal_coordination_geometry_pilot.py:93:        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
```

## docs/notebook_outputs/raw/metal_single_gpu_20h_v2_profile_20260916/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
```

## docs/notebook_outputs/raw/metal_single_gpu_authorized_20260917/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
```

## docs/notebook_outputs/raw/pmm_core_continuation_20260928/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/pmm_five_class_screen_20260928/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/pmm_ion_v2_context_20260926/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
```

## docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
```

## docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/initial_v2_1/screen_agreement_v2_1.md

```text
/media/mechti/Data1/DeepMzyme_worktrees/docs/audit_pmm_screen_agreement.py:409:    (output / "screen_agreement_v2_1.md").write_text("\n".join(lines) + "\n")
```

## docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/screen_agreement_v2_1.md

```text
/media/mechti/Data1/DeepMzyme_worktrees/docs/audit_pmm_screen_agreement.py:409:    (output / "screen_agreement_v2_1.md").write_text("\n".join(lines) + "\n")
```

## docs/notebook_outputs/raw/remote_homology_v1_20260917/README.md

```text
scripts/prepare_colab_smoke_snapshot.py:29:         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
prepare_training_and_test_set/step6_create_additional_split_non_overalpped_structures.py:540:        write_text(staging_dir / "README.md", build_readme(metadata))
prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py:338:    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:37:        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:119:    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py:182:    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
src/build_clean_single_donor_subset.py:828:    (output_root / "README.md").write_text(readme, encoding="utf-8")
prepare_training_and_test_set/provenance/SHA256SUMS:1:ee599e70353872043bf75406020792f6dfe0d5ed6c28080abb24f66a4896841f  common_pdbid_70_30/generated_README.md
prepare_training_and_test_set/provenance/SHA256SUMS:3:8f64ca49a72407bfa2bb4d2b5b638ab9af3bacfea8a2c331db1e53d634656761  exact/generated_README.md
tests/smoke_checks.py:2666:    for relative_path in ("README.md", "docs/archive/workflows/list_train_commands_legacy.md"):
tests/smoke_checks.py:2906:    readme = REPO_ROOT / "bench" / "README.md"
tests/smoke_checks.py:2908:        raise AssertionError("bench/README.md is missing.")
```

## docs/notebook_outputs/raw/remote_homology_v1_20260917/reports/ec/remote_homology_report.md

```text
src/report_remote_homology.py:395:    (args.output_dir / "remote_homology_report.md").write_text("\n".join(summary) + "\n")
```

## docs/notebook_outputs/raw/remote_homology_v1_20260917/reports/metal/remote_homology_report.md

```text
src/report_remote_homology.py:395:    (args.output_dir / "remote_homology_report.md").write_text("\n".join(summary) + "\n")
```

## docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md

```text
src/serial_metal_campaign/reporting.py:464:                    source="docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md"),
```

## docs/plans/metal_level_metal_task_compared_PMM_final_plan.md

```text
scripts/run_metal_5fold_cv.py:1068:            "PMM ion campaign profile (plan: metal_level_metal_task_compared_PMM_final_plan.md). "
```
