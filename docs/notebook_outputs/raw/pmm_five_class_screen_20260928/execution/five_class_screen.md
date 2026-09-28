# Five-class fold-0 screen

Five-class strict replay passes: 2/3.

Exploratory single fold, seed42. Native metrics use different vocabularies and cannot rank formulations. Only common-four deltas are paired. GVP6 control retains supplemental v2.1 status; no new strict receipt. TECH-023/025 full-grid gates remain open. Backup/provider shutdown verification is separate.

Native five-class Co+Ni is exported internally as Class VIII; common-four Class VIII means Fe+Co+Ni. Values below are percentages.

| Family | Target | Status | Epoch | Common4 BA | Common4 F1 | Native BA | Native F1 | Fe recall | Co+Ni recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| only_esm | four_class | strict_replay_certified | 36 | 88.3945 | 84.2951 | 88.3945 | 84.2951 | — | — |
| only_esm | five_class | strict_replay_certified | 41 | 89.4354 | 86.1772 | 80.8663 | 78.8576 | 89.7906 | 45.3947 |
| only_esm | six_class | strict_replay_certified | 33 | 85.8393 | 79.8831 | 72.2507 | 65.8121 | 79.8429 | — |
| only_gvp | four_class | strict_replay_certified | 12 | 86.0232 | 77.2357 | 86.0232 | 77.2357 | — | — |
| only_gvp | five_class | failed_independent_replay | — | — | — | — | — | — | — |
| only_gvp | six_class | historical_supplemental_v2_1_only | 31 | 82.8399 | 75.4985 | 63.6275 | 57.4932 | 75.6545 | — |
| gvp_late_fusion | four_class | strict_replay_certified | 11 | 89.4653 | 80.7283 | 89.4653 | 80.7283 | — | — |
| gvp_late_fusion | five_class | strict_replay_certified | 50 | 87.6608 | 84.4260 | 84.5951 | 83.1475 | 77.4869 | 73.6842 |
| gvp_late_fusion | six_class | strict_replay_certified | 32 | 88.8124 | 85.3215 | 72.0131 | 72.1372 | 85.8639 | — |

Common-four paired deltas (percentage points; no formal significance claim):

- only_esm, five_class minus four_class: BA +1.0409; macro-F1 +1.8821; control strict_replay_certified.
- only_esm, five_class minus six_class: BA +3.5960; macro-F1 +6.2941; control strict_replay_certified.
- gvp_late_fusion, five_class minus four_class: BA -1.8045; macro-F1 +3.6977; control strict_replay_certified.
- gvp_late_fusion, five_class minus six_class: BA -1.1515; macro-F1 -0.8955; control strict_replay_certified.

CSV values are fractions; JSON includes every class recall/support, confusion matrices, identity and input hashes.
