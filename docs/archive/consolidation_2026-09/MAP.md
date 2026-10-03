# Job B history and ownership map

Source commands: `git show 3c0f80c:EXPERIMENT_STATUS.md`,
`git show 9a25f36^:EXPERIMENT_STATUS.md`, `git show 9a069d2^:EXPERIMENT_STATUS.md`,
and `git show b45b893:EXPERIMENT_STATUS.md`.
All four complete texts are preserved as deduplicated verbatim blocks with
source ranges and SHA-256 in [the manifest](inventory/job_b_status_preservation.json).
This also recovers `b45b893:72-86`, the removed 09-16/17 records, Test-use status,
Current blockers, Immediate next action and Update rule. Literal historical
links stay literal in text fences; README evidence links are navigable.

| History | Destination |
|---|---|
| PMM ion, core scopes, replay diagnoses, five-class screen, pause | [PMM README](../campaigns/pmm_ion_metal/README.md), [log](../campaigns/pmm_ion_metal/log.md) |
| Zenodo reconstruction/relaunch | [README](../../campaigns/zenodo_pmm_exact_2026-09-24/README.md), [log](../../campaigns/zenodo_pmm_exact_2026-09-24/log.md) |
| Single-GPU rejected admission and separately paused continuation | [README](../../campaigns/metal_single_gpu/README.md), [log](../../campaigns/metal_single_gpu/log.md) |
| Exact-PMM exploratory pocket benchmark | [README](../campaigns/exact_pmm_5fold_2026-09-23/README.md), [log](../campaigns/exact_pmm_5fold_2026-09-23/log.md) |
| Architecture, geometry and matched RING pilots | [README](../campaigns/metal_pilots_2026-09-15/README.md), [log](../campaigns/metal_pilots_2026-09-15/log.md) |
| Development-only metal × EC1 association | [README](../campaigns/metal_ec1_association_2026-09-15/README.md), [log](../campaigns/metal_ec1_association_2026-09-15/log.md) |
| EC1 standalone reference, including formerly truncated continuation | [README](../campaigns/ec1_standalone_v12_2026-09-14/README.md), [log](../campaigns/ec1_standalone_v12_2026-09-14/log.md) |
| CPU sequence-remoteness validation reuse | [README](../campaigns/remote_homology_v1/README.md), [log](../campaigns/remote_homology_v1/log.md) |
| Cross-campaign policy, tables, old blockers, test ledger and update rules | [Documentation-record README](../campaigns/project_status_2026-09/README.md), [log](../campaigns/project_status_2026-09/log.md) |

The final row is an archival documentation record, not a new scientific campaign.
Closing that record does not close any paused experiment. The pilot folder above
preserves STATUS text only; Job C's full pilot move has not happened.

AGENTS section relocations are in [MOVED](../../MOVED.md); every inventoried
content block has a reviewed disposition in [RULES](RULES.md).
Nothing under `docs/plans/`, raw evidence or existing summaries moves or changes.
No campaign recipe or frozen source identity changes. History retention does
not reinstate old authorizations or erase later qualifications.
