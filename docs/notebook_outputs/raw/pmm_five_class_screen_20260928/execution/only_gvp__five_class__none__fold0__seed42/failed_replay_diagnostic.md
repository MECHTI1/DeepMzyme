# Only-GVP five-class: preserved strict replay failure

**Status: trained_but_strict_replay_failed. All metric point estimates below are provisional.**

50 epochs completed; selected epoch 47. Verified host backup: 70 files. No rerun or tolerance change.

Both exports cover 1492 identical validation ions. Exact nonprobability fields: True. Native/common-four prediction changes: 0/0.

Maximum absolute probability difference: 0.00000238; 22 fields across 6 ions exceed the original 0.000001 threshold.

| Probability column | Maximum absolute difference | Fields > 0.000001 |
|---|---:|---:|
| p_native_Mn | 0.00000224 | 4 |
| p_native_Cu | 0.00000124 | 1 |
| p_native_Zn | 0.00000238 | 4 |
| p_native_Fe | 0.00000114 | 2 |
| p_native_Class_VIII | 8.9E-7 | 0 |
| p_common4_Mn | 0.00000224 | 4 |
| p_common4_Cu | 0.00000124 | 1 |
| p_common4_Zn | 0.00000238 | 4 |
| p_common4_Class_VIII | 0.00000113 | 2 |

| Provisional view | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| native5 | 68.1998% | 65.2286% |
| common4 | 78.2397% | 74.1240% |

native5 recalls: Mn 72.3577%, Cu 87.1795%, Zn 88.9503%, Fe 72.7749%, Co+Ni 19.7368%.

common4 recalls: Mn 72.2222%, Cu 87.1795%, Zn 88.9503%, Class VIII 64.6067%.

**Separate provisional five-vs-six point estimate; excluded from the certified screen comparison:** common-four BA 78.2397% versus 82.8399% (-4.6002 percentage points). GVP5 failed strict replay; historical GVP6 has supplemental v2.1 agreement only. Neither status is upgraded here.

Native Class VIII means Co+Ni. Common-four Class VIII means Fe+Co+Ni, summed before argmax.

The two exports produce identical class-based metrics when predictions match; that observation is not replay certification.

Preserved serialized CSVs only. Identical class predictions do not satisfy strict probability replay and do not certify the run. No input-tensor equality or repeated-forward diagnostic was performed; numerical cause remains unproven. Single fold and seed, no formal inference. The strict screen report must continue excluding this five-class fit.

JSON retains every column's worst UID/value pair, metrics/confusion matrices and source hashes. Canonical checkpoint binaries and full metadata remain in the verified backup.
