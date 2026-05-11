# Biological Metal Chain-Annotation Support Suggestion

This folder contains Codex-suggested alternatives to the step5 biological-metal
verification notebooks. The original notebooks under
`prepare_training_and_test_set/` are intentionally left unchanged.

These suggested notebooks are conservative about interpretation:

- UniProt cofactor annotations are treated as chain/protein-level support for a
  metal type, not as site-level proof that a specific predicted residue pocket is
  native.
- Missing UniProt mapping, failed mapping, failed UniProt fetches, and absent
  cofactor annotations are preserved as unknown annotation, not as non-native
  sites.
- The generated columns use names such as
  `chain_annotation_supports_metal` and `site_native_status` instead of a
  numeric `native` label.
- Unsupported-metal filtering uses `normalized_metaltype`, so FE2/FE3-style
  labels normalize to FE before filtering.
- The output filenames mark the exact PinMyMetal split as auxiliary because
  `Plan.md` treats the non-overlapped PinMyMetal split as the trusted final
  held-out split.

These files are for review before adopting any change into the main preparation
workflow. They do not establish which sites are truly biological without
additional site-level evidence.
