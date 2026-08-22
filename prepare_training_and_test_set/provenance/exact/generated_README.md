# Metal Split PinMyMetal

Available supported-structure projection of the original PinMyMetal train/test membership.

This split preserves the source train/test side for available PDB IDs and may contain train/test PDB-ID overlap. The current DeepMzyme summary CSV does not carry the original PinMyMetal `residueid_ion` / `metalid` row identifiers, so this audit is at the PDB-ID/structure level. It is an exact/possibly-overlapped reference split, not the trusted final held-out split.

Primary CSV: `final_data_summarazing_table_transition_metals_only_catalytic.csv`
Exact train PDB IDs: `1472`
Exact test PDB IDs: `313`
Exact train/test PDB-ID overlap: `177`

See `split_metadata.json` for the source-membership audit.
