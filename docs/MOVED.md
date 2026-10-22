# Documentation relocations

Job B section copies, reviewed by the maintainer on 2026-09-28; Job B moved or renamed
no existing file. Later closures add their own rows.
Source revision: `3c0f80c`. Exact STATUS ranges and hashes are in the
[preservation manifest](archive/consolidation_2026-09/inventory/job_b_status_preservation.json).

| Old location | New owner |
|---|---|
| Former root contributor guide (commit `3c0f80c`), key project files/directories | [Repository navigation](README.md#repository-navigation) |
| Former root contributor guide (commit `3c0f80c`), documentation coordination protocol | [Coordination protocol](README.md#documentation-coordination-protocol) |
| Former root contributor guide (commit `3c0f80c`), metal stage checks and required answer format | [Metal playbook protocol](METAL_TRAINING_PIPELINE_PLAYBOOK.md#stage-request-protocol) |
| `EXPERIMENT_STATUS.md`, dated PMM ion entries | [PMM history](archive/campaigns/pmm_ion_metal/log.md) |
| `docs/campaigns/pmm_ion_metal/` (README, log, storage receipt; closed 2026-10-03, `git mv`) | [Archived PMM campaign](archive/campaigns/pmm_ion_metal/README.md) |
| `docs/campaigns/pmm_ion_metal_v3/` (closed 2026-10-13, `git mv`; the old path stays as a registered alias for frozen tools and records; pinned files and the alias in [frozen_moves.json](archive/frozen_moves.json); original bytes at tag `pmm-ion-metal-v3-final`) | [Archived v3 campaign](archive/campaigns/pmm_ion_metal_v3/README.md) |
| `docs/campaigns/pmm_trunk_lr_2026-10/` (closed 2026-10-13, `git mv`; the old path stays as a registered alias for the frozen trunk-LR tools; the pinned specification and the alias in [frozen_moves.json](archive/frozen_moves.json); original bytes at commit `ad99c5f`, no tag) | [Archived trunk-LR campaign](archive/campaigns/pmm_trunk_lr_2026-10/README.md) |
| Other current/historically removed STATUS sections | [Complete history map](archive/consolidation_2026-09/MAP.md) |
| `docs/BACKUP_AND_DATA_INVENTORY.md` (removed 2026-10-20; maintainer backup details are kept privately) | [Benchmark runs and checkpoint releases](DATASETS.md#benchmark-runs-and-checkpoint-releases) |
| `docs/PMM_AUTHOR_PROVENANCE_REQUEST_DRAFT.md` (unsent email draft; removed 2026-10-20 and kept privately) | [PMM deposition crosswalk](PMM_DEPOSITION_CROSSWALK.md) |
| Former execution-report directory (six reports at commit `041f430`; renamed 2026-10-21 with `git mv`, same file names) | `docs/execution_reports/` |

Existing playbook paths, executable blocks, `docs/plans/`, raw evidence and
existing experiment summaries retain their paths and contents. Job C moves
remain pending their own review checkpoints.
