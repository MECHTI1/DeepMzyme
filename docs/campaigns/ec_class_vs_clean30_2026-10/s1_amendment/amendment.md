# S1 fixed-anchor family selection amendment

- Amendment SHA-256: `d07dbb6a68de7f441a0030661f110f2de9dabd4786a702a1e23d92a3a7a242ea`
- Original Freeze 1: `28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c`
- Rule: `fixed_anchor_holm_v1`; publication is required before execution.

The original freeze documents and prepared data remain unchanged. This amendment changes only S1
family selection and the evidence/authentication needed to execute that rule. Historical selection
remains available for replay. Simulation describes the fixed rule; it does not tune alpha or establish
real-model superiority. R2 scoring and all secondary multiplicity settings remain unchanged.

## Fixed policy

```json
{
  "anchor_family": "F2",
  "candidate_families": [
    "F3",
    "F5",
    "F4"
  ],
  "control_families": [
    "F1",
    "C-G"
  ],
  "fallback_family": "F2",
  "family_alpha": 0.04,
  "family_tie_band_points": 0.5,
  "family_tie_preference": [
    "F5",
    "F3",
    "F4"
  ],
  "family_tie_rule": "preferred passing family within the band of the maximum passing score",
  "gain_and_recall_reference": "F2",
  "historical_rule_id": "family_escalation_v1",
  "incremental_superiority_claim": false,
  "learning_rate_tie_band_points": 0.5,
  "learning_rate_tie_rule": "first listed grid setting within the band of the maximum",
  "multiplicity": "Holm step-down over all three fixed contrasts",
  "promotion_config": {
    "alpha": 0.04,
    "bootstrap_seed": 20261023,
    "guard_min_training_proteins": 30,
    "guard_net_proteins": 3.0,
    "guard_recall_drop_points": 5.0,
    "max_skip_fraction": 0.01,
    "min_gain_points": 1.0,
    "n_draws": 10000,
    "stratified": false,
    "zero_recall_unit": "seed_mean"
  },
  "rule_id": "fixed_anchor_holm_v1"
}
```

## Evidence

| Role | File | SHA-256 |
|---|---|---|
| simulation_spec | `docs/campaigns/ec_class_vs_clean30_2026-10/s1_amendment/simulation_spec.json` | `bfaff6799a65604761a9d1f93736feabf1998e29d48dfa1476e471cc24fed39d` |
| simulation_summary | `docs/campaigns/ec_class_vs_clean30_2026-10/s1_amendment/simulation_summary.json` | `d765f615d61d7d51b2c2b27855a3fba52c710e640c49b6c195be58695cee6624` |
| verification | `docs/campaigns/ec_class_vs_clean30_2026-10/s1_amendment/verification.json` | `3b2053583770372b9b26da845881eaf4da9c2e4e7e4d779248d65129bb0f083f` |
