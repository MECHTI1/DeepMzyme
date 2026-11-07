# Amendment 2: interface contract for the general chain verifier

**Status: draft (2026-11-04), part of the [amendment-2](amendment_draft.md) review package.** This contract fixes
what the general amendment-chain verifier must check and return, and the names that the science files and the
execution v5 files use for each other. Its machine-readable form is
`tests/fixtures/ec_clean30_amendment2_contract.json`. The fixture and this text must agree; a change to either is a
change of the contract and needs both sides' tests to be updated in the same change.

## Ownership

| Part | Written by | Set (rule A2-GOV-2) |
|---|---|---|
| `src/ec_clean30/amendment_rules.py` (shared rules), `src/ec_clean30/amendment_chain.py` (this verifier), the contract fixture, the record builder `src/ec_clean30/amendment2.py` and every other file of the [scope list](amendment_draft.json) | science side | scientific |
| `src/ec_clean30/runtime_precision.py`, the CUDA path of `scripts/run_clean_split30_comparator.py` | execution v5 | scientific, pinned in amendment 2 |
| the declared closed list `execution_file_set.declared` of the [draft record](amendment_draft.json): `phase3.py`, `phase4.py`, `workstation.py`, `runpod_bootstrap.py`, `runpod_controller.py`, `container_caps.py`, `execution_v5.py`, `hot_storage.py` and the three runner scripts of phases 3 and 4 and the workstation; `execution_file_set.pending` is empty | execution v5 | execution (closed list, manifest v5) |

Neither side edits the other's files, with one delegated line: the execution-v5 commit that adds the comparator's
module-level `runtime_precision` import also adds `"src/ec_clean30/runtime_precision.py"` to
`scoring.SCORER_CODE_PATHS`, so that the scorer pin test stays green.

## Entry points

```text
ec_clean30.amendment_chain.verify_chain(tip_path, *, expected_freeze1_sha256, expected_freeze1b_sha256,
                                        repo, require_published=True, expected_chain=None) -> dict
ec_clean30.amendment_chain.check_chain_receipt(receipt_path, *, repo, expected_chain=None) -> dict
```

Both return the keys `links`, `chain`, `fit_chain`, `tip_sequence`, `composite_scientific_pins`, `analysis_pins`,
`execution_file_set`, `post_fit_analysis_files`, `pinned_execution_functions`, `pinned_analysis_functions`,
`policies`, `plan_records`, `dry_run_evidence`, `scientific_identity_sha256`, `published`.

- The module is importable without torch and imports only `amendment_rules`, `freeze` and `provenance` from the
  campaign package.
- The partition sets, function-closure pins, change-class rules, role table, link paths and log lines,
  `r_cmd_v2_check` and `scientific_identity` live in `amendment_rules`; the verifier calls them and does not
  reimplement them.
- `freeze.verify_s1_amendment` is never called on the live tree; it compares the whole live tree with the Freeze-1
  pins plus the S1 changes and therefore fails after any amendment-2 edit.
- **Full clone only.** `verify_chain` runs on the workstation's full clone, the checkout that also holds the private
  files at their bound paths (a fresh clone has none and is refused by C8 and C11). It refuses a partial clone
  (promisor remote, partial-clone filter or promisor packs, also behind an alternate object store) before any other
  check, and its message names `check_chain_receipt`.
- **Foreign code.** Called with a `repo` whose verifier is not the caller's own file, `verify_chain` runs that
  repository's verifier in a fresh interpreter; with `require_published` it does so only when the four verifier
  files (`amendment_chain.py`, `amendment_rules.py`, `freeze.py`, `provenance.py`) are byte-identical to the
  caller's. A real verification therefore always runs inside the verified repository. Even in draft review, a
  delegated result of two or more links without a scientific identity is refused.
- **Scientific identity.** For a tip of sequence 2 or higher `verify_chain`, `check_chain_receipt`, the selector and
  the scorer refuse a missing or malformed `scientific_identity_sha256`.
- **Environment and time.** Every git call and fresh interpreter of a verification runs without the caller's
  `GIT_*` variables, with `GIT_CONFIG_NOSYSTEM=1`, `GIT_CONFIG_GLOBAL` set to the null device and terminal
  credential requests switched off, in its own process group, under one budget of 3600 s for the whole call; each nesting
  level hands its children 20 s less. A timed-out group is killed with its grandchildren, and a child's temporary
  files live in a folder of its parent that is removed whatever happens. A malformed depth or deadline variable is
  refused.
- **Private bindings.** The repository is public, so the files of the private roles (see
  [Private bindings](#private-bindings)) are never committed. `verify_chain` reads them from the working tree of the
  repository it verifies, never from git; a nested verification in a checkout of a link's commit, which holds no
  private file, reads them from the outermost verification repository, whose root the parent passes as an argument
  of the fixed chain probe (an absolute directory, or the nested verification is refused). No environment variable
  names that root, so a top-level call always reads the working tree of the repository it verifies.

## Chain receipts (workstation, pod and per unit)

- **Written once on the workstation.** `amendment2.write_chain_receipt` (runner action `verify-amendment-chain`,
  format `deepmzyme_ec_clean30_amendment_chain_receipt_v1`, schema 2) calls `amendment_chain.chain_receipt_record`,
  which runs `verify_chain` on the full clone and records:
  - the key: the tip's SHA-256, HEAD, the SHA-256 of `git status --porcelain=v1 --untracked-files=all`,
    `origin/main`, the SHA-256 of the four verifier files, `require_published` and `expected_chain`; the key is taken
    before and after the verification, and a difference refuses the receipt;
  - the full `verify_chain` result, the SHA-256 of every link file and rendering, every public file the tip binds
    (`bindings`) and, apart, the private bindings that `verify_chain` checked (`private_bindings`: where, path and
    SHA-256 only);
  - `receipt_sha256`, the SHA-256 of the canonical receipt without that field.
- **Checked on a pod and per unit.** `check_chain_receipt(receipt_path, *, repo, expected_chain=None)` returns the
  result read from the receipt after a cheap re-check of `repo`:
  - HEAD equals the receipt's HEAD, or is a descendant of it, published on `origin/main`, whose changes since the
    receipt touch only execution files;
  - the tree is clean, every link file and rendering has its receipt bytes, the receipt's public and private
    bindings are exactly those of the tip link file, and every public file the tip binds hashes; private bindings
    are never read (a pod holds none of them);
  - the live scientific and post-fit analysis pins hold and the pinned execution functions recompute to their pins;
  - the verifier files equal those of the receipt, `receipt_sha256` holds and `expected_chain`, when given, equals
    the chain.
- It works in a `git clone --filter=blob:none` partial clone (it reads only checked-out files, commits and trees),
  also when the promisor is unreachable. A receipt made with `require_published=False` (`--draft`) is refused.
  Measured on 2026-11-04 on the real tree's file set (239 tracked `src/` and `scripts/` files, the nine
  execution-v4 files, 25 bound files) under the two-core cap: 47 to 53 ms per check, up to 72 ms while the machine
  was otherwise busy. A full `verify_chain` of the
  two-link chain from a later HEAD stays a once-per-session workstation step.

## Links and chains

| Sequence | Record | Format | SHA-256 | Source commit | Authenticated by |
|---|---|---|---|---|---|
| 1 | `s1_amendment/amendment.json` | `deepmzyme_ec_clean30_s1_amendment_v1` | `d07dbb6a68de7f441a0030661f110f2de9dabd4786a702a1e23d92a3a7a242ea` | `3154a7f3b0ad44e8dffbc66072990cfb7d9c9030` | unchanged `freeze.verify_s1_amendment`, run in a checkout of that commit |
| 2 | `amendment2/amendment.json` (not yet written) | `deepmzyme_ec_clean30_amendment_v2` | set when the record is built | the publishing commit | `amendment_chain.verify_chain` as published with link 2, run in a checkout of that commit |
| n >= 3 | `amendment<n>/amendment.json` | `deepmzyme_ec_clean30_amendment_v2` | set when the link is built | its publishing commit | `amendment_chain.verify_chain` as published with link n |

- **Chain:** the SHA-256 values of all links in sequence order.
- **Tip:** the last link. Only the tip is compared with the live tree.
- **Fit chain:** the chain up to its last link that changed a scientific file. Fit, step, futility and unit identities
  carry the fit chain; a class-(a2) analysis link extends the chain but not the fit chain.
- **Paths.** One fixed path per sequence (`amendment_rules.link_path`); the rendering `amendment.md` and the reviewed
  document `amendment_draft.md`, whose rule text the link binds, sit beside it.
- **Publication line.** The campaign log publishes a link of sequence n >= 2 with exactly one line
  `Amendment chain link <n>: ` followed by the SHA-256 in backticks (an optional leading `- `). The link file, its
  rendering and this line are committed together: the commit that introduced the line is the link's source commit,
  and no parent of it holds the link file. Link 1 keeps its historical log entry, which must hold its SHA-256.
- **Written once.** On the history reachable from HEAD every link file was added once with its published bytes and
  never modified or deleted, also when a rewrite is later restored.
- **Log at HEAD.** It names each link of sequence 2 to the tip once and no other: no second line of one sequence (a
  fork or a rewrite), no line beyond the tip (a stale tip), no other line holding the marker.
- **Per-link authentication:** each link is checked in a detached sparse checkout of its own published source
  commit, following the existing `phase3._baseline_authentication` pattern. Before the code published there runs,
  every `src/` and `scripts/` blob of that commit is hashed and must equal the composite pins of the authenticated
  records. The nested result is cross-checked against the records (fit chain, pins, execution set, plans, dry-run
  evidence, policies) and must carry a scientific identity that recomputes.

## Checks

A failure of any check refuses the chain; nothing is downgraded to a warning.

| ID | Check | Refused, for example |
|---|---|---|
| C1 | **Link format.** Each link has its declared format, `schema_version` 1, the campaign ID, its sequence, its fixed path and, from link 2 on, every key of the fixture's `amendment_record.required_keys` and `record_shape.built_keys` (see [Record shape](#record-shape)); only link 2 has the class `pre_data`. | A missing key; another campaign ID; an unknown format; a link at another path. |
| C2 | **Canonical bytes.** Each link file equals the canonical serialisation of its own builder, and its rendering equals the builder's rendering of it. | Re-indented JSON; an edited rendering. |
| C3 | **Order and history.** Sequence numbers start at 1 and increase by 1; `previous_sha256` of link n is the SHA-256 of link n−1; `chain_before` equals the earlier links; `expected_chain`, when given, equals the chain; every link file is written once; the log at HEAD publishes exactly this chain. | A fork, gap, reorder, second link of one sequence, rewritten link or stale tip. |
| C4 | **Self-published, anchored verifier.** Each link is authenticated by the verifier code published at its own source commit, run in a checkout of that commit, after that code is matched with the records' pins; its result is cross-checked. | A link that verifies only on the live tree; a lenient verifier at the link's commit. |
| C5 | **Scope.** `source_changes` equals the set of tracked `src/` and `scripts/` files whose bytes differ between the previous composite pins and this link's pins; each `before` equals the previous composite pin (null for a new file); no pinned scientific file is deleted; no execution file is in scope. | An unlisted changed scientific file; an execution file in scope; a deletion. |
| C6 | **Live source (tip only).** Every scientific file equals the composite scientific pins and post-fit analysis files the analysis pins, except within a class-(a2) link's limits; the pinned function closures equal their pins. That the execution files equal the execution manifest in force is checked by manifest v5, not by this verifier. | A scientific file edited after publication. |
| C7 | **Partition and import isolation.** The execution file set is closed, sorted, disjoint from the scientific and analysis sets and free of forbidden roles (`amendment_rules.partition`); a fresh interpreter that imports `campaign`, `commands`, `views`, `provenance`, `identity`, `selection`, `selector`, `stats`, `futility`, `clean_sanity`, `amendment_rules`, `amendment_chain` and `runtime_precision` loads no execution file, no post-fit analysis file and no untracked file; the static import closure of no scientific or analysis file (the trainer and embedding modules included) holds an execution file; analysis code is reached only through the pinned channels and no execution file imports it; no execution file reaches a private name of a scientific or analysis module. | An execution file that imports, reads or sets a private name of a scientific module (absolute or relative import, module attribute, `getattr`/`setattr`, `importlib`); a scientific module that loads an execution file. |
| C8 | **Evidence.** Every public bound file (decisions, verification, futility operating characteristics, plans, fixtures, the attestation) hashes to its recorded SHA-256 at the link's commit; every private binding (dry-run receipts, tie exposure, F6 inputs) hashes in the verification repository's working tree, follows the data-root link rule (see [private bindings](#private-bindings)), lies below the private root and is not tracked at the link's commit; every binding has the privacy of its role (`amendment_rules.binding_privacy_problems`); every role of the [role table](#required-bindings) is bound; the decisions file itself settles every confirmation, has no placeholder token and no absolute path and equals the record's `decisions_file`; the rule text hashes and is found exactly once in the reviewed document `amendment<n>/amendment_draft.md`. With HEAD past the tip's commit, the tip's bound files also hash on the live tree and no private file is tracked there. | A changed evidence file; a role still holding a note; a private input bound publicly, committed, altered, absent or behind a link; a public binding flagged private; an unsettled decisions file; rule text that is not the reviewed document's. |
| C9 | **Policies.** Each `rules` entry names its registered function; every rule policy, the analysis rule included, equals the canonical output of its registered function at the link's commit, recomputed in a fresh interpreter so that no scientific process loads analysis code. | A tampered policy; copied policy hashes over a redefined rule. |
| C10 | **Plans.** The five v1 plan SHA-256 values equal Freeze 1; the five v2 plan records equal `campaign.build_fold_plan_v2` at the tip. | Any v1 or v2 plan drift. |
| C11 | **Dry runs.** Five dry-run-v2 receipts are bound privately, are read from the verification repository's working tree (never from git), are not tracked at the link's commit, exist, hash, passed, and ran on a clean tree at the recorded source commit on the fold view whose manifest SHA-256 Freeze 1 records; the Freeze-1 in-view receipts are byte-unchanged; `r_cmd_v2_check` passes from that commit to the tip. The result's `dry_run_evidence` carries each fold's receipt binding with that view manifest SHA-256 (a digest, never a count); the per-unit view binding (`phase3._bind_view`) requires the fold's amendment-2 dry-run evidence, compares the view's manifest SHA-256 with it and hashes the in-view receipt it reads, and never reads the private receipt. | A missing, altered, publicly bound or committed receipt; a receipt of a dirty tree or of another view; a rewritten in-view receipt; a protected-file change after the dry runs. |
| C12 | **Publication.** With `require_published`, each link's source commit is an ancestor of `origin/main` and its log publishes the link; HEAD is published. | An unpublished link; a log without the line. |
| C13 | **Data state.** Link 2 records an empty fold-0 record list and zero completed G2 units. A link of sequence 3 or higher carries its change class, timestamps and fold-0 record list and passes `amendment_rules.check_change_class` (for a defect, every defect field; for classes execution and analysis completion, the previous link's pinned execution and analysis functions). A `FUT` entry binds its pointer file and futility record as repository files (see [Record shape](#record-shape)); a FUT record is never taken from the link's own statement. | A later link without its class or fold-0 records; a defect amendment after a bound `stop_futility`; an execution link that re-pins `lane_gate`. |
| C14 | **Scientific identity.** `amendment_rules.scientific_identity` over exactly `SCIENTIFIC_IDENTITY_KEYS`; no execution pin, manifest hash, commit, provider, hardware or session field, also inside the deferred-input specifications. A chain of two or more links always has one. | A record holding a commit or a manifest hash. |
| C15 | **Precedence.** Every record accepted under a chain carries that chain's fit chain and was made at the commit that published the chain's last scientific link or at a descendant of it (`git_head` or `provenance.git.head`), and that commit is published; its `created_utc` is a secondary check. A record bound to a shorter prefix is accepted only in replay mode and is never mixed with full-chain records. | A G2 record or fit made at the dry-run commit, presented under the two-link chain. |
| C16 | **Analysis completion (class a2).** A link of sequence 3 or higher with `change_class` `analysis_completion` changes only the four post-fit analysis files, is published after a bound fold-0 FUT record and before Freeze 2 (`freeze2.json` in the campaign folder), leaves `pinned_analysis_functions`, the `freeze.live_constants()` SHA-256, `scientific_identity_sha256`, `fit_chain` and every rule policy unchanged, and passes `amendment_rules.check_analysis_completion`. At its commit every test the bound refusal fixture names and the decision-fixture test pass in a fresh pytest, and every test the previous link's fixture names keeps its function-closure SHA-256. | An analysis link before FUT; one that also edits `stats.py`; one that changes a decision or weakens a named test. |

## Record shape

The fixture's `amendment_record.record_shape` fixes how a link of sequence 2 or higher holds its fields, including
the fields that `required_keys` names without shaping them. The record builder writes this shape and the verifier
reads it; both are tested against the fixture, and neither copies the other's values.

| Field | Shape |
|---|---|
| `campaign_id` | the campaign ID (C1; carried beside `required_keys`) |
| `base_freeze1` | `{path, sha256}` of the original Freeze 1 |
| `base_amendment` | `{sequence, amendment_id, path, sha256, commit}` of the previous link |
| `execution_file_set` | `{files, names_sha256}`: the sorted execution file list and the SHA-256 of that list |
| `pinned_execution_functions`, `pinned_analysis_functions` | `"<path>::<name>"` -> closure SHA-256 (`amendment_rules.function_closure_sha256`) |
| `plans` | `v1_plan_sha256` and `v2_plan_sha256`, each fold `0`-`4` -> plan SHA-256 |
| `dry_runs` | `rule` (`r_cmd_v2`), `source_commit` (full commit), `receipts` (fold -> `{path, sha256, private: true}`), `in_view_receipts_unchanged` (fold -> SHA-256), `v1_resolved_config` (`{path, sha256}`) |
| `source_changes` | path -> `{before, after}` (`before` is null for a new file) |
| `unchanged_scientific_files` | a list of repository paths; a folder ends in `/` |
| `data_state` | link 2: `fold0_records_at_draft` = `[]`, `g2_completed_units` = `0` and `attestation_at_build` (`{path, sha256}`) |
| `evidence` | `futility_oc`, `interface_contract`, `power_simulation` and `verification`, each `{path, sha256}` |
| `inputs` | `clean_tie_exposure` (`{path, sha256, private: true}`), `f6_holdout` (fold bindings) and `f6_corpus` (`{tables, union_fasta}`), every file bound privately |
| `fixtures` | `refusals` and `decisions`, each `{path, sha256}` |
| `rules` | rule ID -> `{policy_function, analysis}` for every registered rule and `r_cmd_v2` |
| `rule_policy_sha256` | rule ID -> SHA-256 of its canonical policy record |
| `decisions_file` | `{sha256, maintainer, answers}` |
| `bound_files` | name -> `{path, sha256}`; `bound_files.decisions` is the decisions file |
| `confirmation_status`, `provisional_markers_settled`, `publication` | `confirmed`, `true`, `required_before_execution` |
| `rule_text` | `{source, sha256, text}`; `source` is the repository path of the reviewed document |
| `fold0_records` (sequence 3 or higher) | a list of `{kind, sha256}`; a fold-0 futility record has `kind` `FUT`, its `decision` and `pointer` and `record`, each `{path, sha256}` |

A file binding is any `{path, sha256}` object below `bound_files`, `fixtures`, `evidence`, `inputs`, `plans`,
`dry_runs` or `data_state` (`record_shape.file_binding_sections`); C8 hashes each public one at the link's commit and
each private one in the verification repository's working tree.

### Required bindings

The fixture's `record_shape.required_bindings` is the role table (`amendment_rules.REQUIRED_BINDINGS`). Every link of
sequence 2 or higher binds each role by SHA-256: `data_state.attestation_at_build`, `dry_runs.v1_resolved_config`,
`evidence.futility_oc`, `evidence.interface_contract`, `evidence.power_simulation` and `evidence.verification` (one
file each), `inputs.clean_tie_exposure` (one file), `inputs.f6_holdout` (folds `0` to `4`, each one file or an
object of files) and `inputs.f6_corpus` (`tables` by fold and `union_fasta`). The last three are private roles.

### Private bindings

The repository is public. Some inputs hold per-fold counts, accessions with their EC classes, training sequences,
absolute paths or host names; five per-fold counts would give the test-side counts by subtraction, which the
campaign does not publish before Stage 7. `amendment_rules.PRIVATE_ROLES` (fixture
`record_shape.required_bindings.private_roles`) names them: `dry_runs.receipts` (the five dry-run-v2 receipts),
`inputs.clean_tie_exposure`, `inputs.f6_holdout` (every fold entry and every file of an object of files) and
`inputs.f6_corpus` (every table file and `union_fasta`).

- **Shape.** Every binding below a private role is exactly `{path, sha256, private: true}`: no other key, so no
  private content enters the record. `path` is a safe repository-relative POSIX path below the gitignored campaign
  data root `amendment_rules.PRIVATE_BINDING_ROOT` = `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/`; the
  files live in its `amendment2/` folder. Every other binding is public `{path, sha256}`, never carries `private`,
  never lies below that root and is committed with the link.
- **Data-root links.** The campaign data may live on another disk, so a symbolic link is allowed only on a
  component of the data root itself, `amendment_rules.PRIVATE_ROOT_LINK_COMPONENTS` = `DeepMzyme_Data`,
  `DeepMzyme_Data/campaigns` and `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10` (derived from
  `PRIVATE_BINDING_ROOT`). Such a link points to an existing directory outside the repository (by its real path), so
  a private file is never a repository file reached by another name; without a link the file resolves to itself
  inside the repository. Every folder below the campaign folder (`amendment2/` and its subfolders) and the file
  itself are real, never links. One shared helper, `amendment_rules.private_link_problem`, holds the rule for the
  builder and the verifier, with one message each for a link below the campaign folder, a data-root link into the
  repository and a dangling or non-directory data-root link.
- **Builder.** A private file is a regular file below the private root inside the repository (lexically), reached
  by the data-root link rule, not tracked by git (index or HEAD) and ignored by a pattern of a `.gitignore` that is
  tracked at HEAD, unmodified and in a folder that contains the file (`git check-ignore -v`, with system, global and
  repository `core.excludesFile` off; a match in `.git/info/exclude` or a negated pattern is refused, since a local
  exclude holds in one clone only; a path through a data-root link is asked at that link, since git does not look
  past a symbolic link). The full path must also be ignored as a clone of HEAD with real folders sees it: in a
  throwaway repository without `.git/info/exclude` that holds the HEAD `.gitignore` of every folder containing the
  file, and the file as real folders and an empty file, git must ignore it by a non-negated pattern of one of those
  files (a folder-only negation such as `!DeepMzyme_Data/campaigns/` never matches a link, yet re-includes the real
  folder of another clone, so it is refused). A dry-run
  receipt must also name the fold view whose manifest SHA-256 Freeze 1 records. Only its path and SHA-256 are
  recorded; the rendering
  `amendment.md` lists them in a table of private bindings and nothing else of them.
- **Verifier.** C8 and C11 read private files from the verification repository's working tree, never from git, and
  refuse one that is absent, does not hash, breaks the data-root link rule, lies outside the private root or is
  tracked at the link's commit; a private role bound publicly and a public binding flagged private are refused. The
  C11 content checks run on the private receipt bytes unchanged.
- **Execution side.** It matches these files by digest only: `phase3._bound_pretrain_file` and `phase4` compare a
  file's SHA-256 with `amendment_rules.file_bindings(binding_at(tip, role))` (a host that runs these checks holds
  the private F6 and tie files it uses at their bound paths), and `phase3._bind_view` compares the view's manifest
  SHA-256 with the fold's `fold_dry_run_evidence` (C11 checked the private receipt on the workstation). A pod, a
  partial clone of `origin` that holds no private file, checks the chain receipt and binds its views without the
  dry-run-v2 receipts.

### Bound futility files

A `FUT` entry binds the binding pointer file `fut_seq<k>.json` (format `deepmzyme_ec_clean30_futility_pointer_v1`,
binding, k from 2 to below the link's sequence, a fit chain that is a prefix of the chain before the link, the
futility record's SHA-256) and the futility record it names (format `deepmzyme_ec_clean30_futility_record_v1`, the
same decision, binding, the same fit chain), as repository files committed with the link.

## Record builder

`amendment2.build_amendment2` writes link 2 once (fixture section `record_builder`).

- **From the reviewed draft:** every key of `record_builder.draft_keys` with its listed sub-keys. The pinned-function
  entries of the draft name functions (path -> top-level names); these names must equal
  `amendment_rules.PINNED_EXECUTION_FUNCTIONS` and `PINNED_ANALYSIS_FUNCTIONS`. `status_note` and
  `source_changes_pending` are dropped.
- **Two documents, two roles:** the rule text is read from the reviewed document `amendment2/amendment_draft.md`
  (keyword `amendment_draft_md`, any other document is refused); the rendering is written once to
  `amendment2/amendment.md`.
- **Computed at build from the live tree** (values that depend on execution-side code that is still changing):
  - the execution file set: every tracked `src/` and `scripts/` file that neither the composite pins before link 2
    (Freeze 1 and link 1) nor `source_changes` name, checked by `amendment_rules.partition`; it must equal the
    draft's declared closed list `execution_file_set.declared`, and `execution_file_set.pending` must be empty;
  - the closure SHA-256 of every pinned function (`phase3.lane_gate` now; `phase4.replay_cpu_logits` once its source
    exists and the registry names it);
  - `live_constants_sha256`, `rule_policy_sha256` and `selection_policy_sha256_v2`.
- **Build inputs:** the five plan-v2 records (`plan-v2`; each equals `campaign.build_fold_plan_v2` live) fill
  `plans.v2_plan_sha256`; the five dry-run-v2 receipts, written once below the private root and bound privately, fill
  `dry_runs.receipts`, and their common clean HEAD fills `dry_runs.source_commit`; every required role is bound with
  `--bind ROLE=PATH` (a dotted name, or a key below one such as `inputs.f6_holdout.0` or
  `inputs.f6_corpus.union_fasta`), privately below a private role. The builder never fabricates them.
- **Refused, in this order:** another draft format; a draft without its contract shape; an invalid decisions file;
  missing build inputs; a pending source change; a pending execution file; base documents that do not hash; a scope
  list that differs from the live tree, or a pinned file changed outside it; a derived execution set with a
  scientific or post-fit analysis role, or other than the declared closed list; pinned-function names that differ
  from the registries; build inputs that do not check (a private file outside the private root or the repository,
  behind a link that the data-root link rule refuses, tracked or not ignored by git, or a dry-run receipt of a dirty tree included); a rule-text
  document other than `amendment_draft.md`; a required binding that is absent or does not have the privacy of its
  role; file bindings that do not hash. Nothing is written on a refusal, and an existing record is never rewritten.

## Scientific identity and execution manifests

- The scientific identity holds the campaign, Freeze-1 and Freeze-1b hashes, the fit chain, the composite scientific
  pins, the pinned execution functions, the v1 and v2 plan hashes, the Freeze-1 view records, the amendment-2 dry-run
  receipts, the precision policy and the deferred-input specifications.
- G2 measurements, Freeze 1c v2, the numerical G3 authority, unit results, step records and the futility record
  compare `scientific_identity_sha256` and `fit_chain`, not the exact execution-manifest SHA-256. A later execution
  manifest is accepted for them only if it is published, verifies and has the same scientific identity.
- Execution manifest v5 binds the two-link chain and the scientific identity and refuses real work when the chain tip
  is below sequence 2. Manifests v1 to v4 stay verifiable for their historical records only.

## Pinned execution functions and precision

- `phase3.lane_gate` (G3) and the CPU replay comparison, proposed as `phase4.replay_cpu_logits`, are pinned by
  function-closure SHA-256 (`amendment_rules.function_closure_sha256`). The replay compares on CPU at an absolute
  tolerance of 1e-5, checks near ties at 2e-5 and requires argmax agreement. An execution or analysis-completion
  link keeps both registries' pins of its previous link.
- `runtime_precision` provides `configure_and_record`, `validate_policy`, `read_policy_log`,
  `execution_environment` and `bootstrap_code` (the comparator builds its child prelude with it at import). TF32 is
  off for every GPU fit, embedding, CLEAN fit, export and farm child; scientific verifiers refuse a receipt whose
  recorded policy differs.

## Consumers of the verifier

- **Selector dispatch** on the fit chain of `selector._verify_amendment_chain`: a real call runs `verify_chain` once per
  process and re-checks its in-memory receipt (`check_chain_receipt_record`) on later calls, or checks a
  `--amendment-chain-receipt` with `check_chain_receipt`; `--smoke` verifies without `require_published` and never
  takes a receipt. An empty chain gives the historical escalation; the S1-amendment chain alone gives the fixed-anchor
  rule for replay or smoke only; the two-link chain with a scientific identity gives amendment 2. A tip with extra
  class-(a2) links is accepted only after a verified fold-0 FUT record.
- **Futility gate:** `futility.verify_futility_decision` (also `--amendment-chain-receipt`) reads the exact-name
  pointer files `amendment2/futility/fut_seqN.json` below the campaign root, N from 2 to the tip sequence, with no
  glob. Any `stop_futility` pointer wins. Fold-0 S4 units, every fold 1-4 unit and every final-arm CLEAN fit are
  refused without a verified `continue`. A caller that authenticated the chain passes it as `expected_chain`.
- **Refit and folds 1-4:** `selector.verified_refit_recipe` checks the S4 v2 record against the fit chain and, in
  fold 0, FUT = `continue` under the same chain; `refit-recipe` authenticates once and expects that chain afterwards;
  `campaign.fold_units_v2` returns the S1 or S4-template units.
- **Scoring:** the scorer imports neither `amendment_chain` nor `clean_sanity`. It reads a `verify-amendment-chain`
  receipt and the CLEAN sanity receipts as JSON, bound by SHA-256 in the scoring manifest, and refuses a draft
  receipt.
- **Execution side (manifest v5, per unit):** `check_chain_receipt` on the receipt written on the workstation,
  with `expected_chain` from the manifest; never `verify_chain` on a pod. Records under a chain are checked with
  `record_precedence_problems(verified, records, repo=...)`.

## Choices recorded by the hardening review (2026-11-04)

Where the review offered a design choice that the rule text leaves open, the stricter option was taken:

- one fixed path per sequence and a strict log line per link, committed with the link (A2-GOV-1(3));
- foreign verifier code never judges a real chain, instead of a byte comparison of the verifier file alone;
- a FUT record counts only through bound pointer and record files, and class (a2) reruns the named tests and pins
  their function closures, since test files are not pinned otherwise;
- every rule policy is recomputed at each link, not only the scientific ones;
- private names are refused through any import form and any module attribute, not only `from` imports;
- the import closure of every scientific and analysis file is checked, not only the trainer entry points;
- execution files may not import post-fit analysis modules;
- the files a published tip binds may not change on the live tree, although `r_cmd_v2` leaves `docs/` and `tests/`
  free;
- `runtime_precision.py` stays a scientific file (pending as a source change), never a pending execution file;
- Freeze 2 of this campaign is the document `freeze2.json` in the campaign folder.

The private-name rule holds on the live tree: the declared execution files reach no private name of a scientific or
post-fit analysis module.

## Tests that prove the contract

- `tests/test_ec_clean30_amendment_rules.py` (science side, staged): closure pins, partition refusals, private names
  through every import form, link paths and log lines, the role table, change classes with function pins and
  `r_cmd_v2` on the fixture links.
- `tests/test_ec_clean30_amendment_chain.py` (science side, staged): the synthetic published repository (execution
  commits and a merge between link 1 and amendment 2, filled evidence, inputs and required bindings, synthetic
  refusal and decision-fixture tests); link 1 and link 2 at their commits; refusal of a fork, gap, reorder,
  unpublished link or rendering drift; partition, import isolation and the import closure; live source; scientific
  identity; records bound by git ancestry and publication (`amendment_chain.record_precedence_problems`); the fold's
  amendment-2 dry-run evidence (`amendment_chain.fold_dry_run_evidence`).
- `tests/test_ec_clean30_amendment_chain_attacks.py` (science side, staged): rewritten, restored and early-logged
  links; a predecessor named at another commit; link 1 and a lenient verifier checked against their pins at their own
  commits; foreign verifier code; the cross-check; a pod partial clone, with and without its promisor, and every
  receipt refusal, a receipt race and a bound file changed after the receipt; a run-time load of an execution file; exported `GIT_*` variables; forks and stale tips; class (a2) with
  bound futility files, policies, decisions, named tests, Freeze 2 and scope; a defect after a stop; a re-pinned gate;
  private names and analysis imports of execution files; C5, C6, C8, C9, C11 and C13 at link 2's commit (scope,
  gate pins, roles, decisions, rule text, policies, live constants, dry runs, data state, source parent); bound files
  on the live tree; time budgets, process groups and malformed variables; and a sequence-3 link of each class, accepted from
  a later HEAD and through its receipt in a partial clone.
- `tests/test_ec_clean30_amendment_chain_private.py` (science side, staged): private bindings; the shared shape
  rules; the builder's refusals (a public or committed copy, a path outside the root or the repository, a symbolic
  link below the campaign folder, a tracked or unignored file, a dirty dry-run receipt, a private binding in the
  draft) and its rendering of paths and digests only; the verifier's refusals at link 2's commit and from a later HEAD (a private role bound
  publicly, a public role flagged private, a path outside the root, with `..` or absolute, a private file tracked at
  the link's commit or later, altered, absent or behind a link below the campaign folder, an altered or dirty
  dry-run receipt); the data-root link rule on the workstation layout (a real, gitignored `DeepMzyme_Data/` whose
  `campaigns` entry links to a folder outside the repository), built and verified end to end in place, from a later
  HEAD and through the chain receipt on a pod, and in builder and verifier alike: data-root links outside the
  repository accepted (the whole `DeepMzyme_Data` folder included), a data-root link into the repository, a
  dangling or non-directory one, and a link at or below the `amendment2/` folder or on the file itself refused with
  their exact messages; a top-level
  call that exports a nesting depth and a private root naming another working tree; a root ignored only by a local
  exclude or by an uncommitted `.gitignore` change; on the workstation layout a committed `.gitignore` under which
  a clone with real folders does not ignore the private file (a folder-only negation of `DeepMzyme_Data/` or
  `DeepMzyme_Data/campaigns/`) refused by the builder, an unrelated negation accepted; the chain receipt checked in a pod without the private files; the
  digest match of the execution side and the view binding on a pod clone without the dry-run-v2 receipts.
- `tests/test_ec_clean30_amendment2_build_chain.py` (science side, staged): link 2 built by the real
  `build_amendment2` on the same synthetic published repository, rendered, published and accepted by `verify_chain`;
  a published builder output without one required key refused; both sides' names equal to the fixture's record shape;
  the builder's refusals (required bindings, declared execution list, pending execution files, rule-text document);
  the staged review draft refused only for its missing build inputs.
- Execution v5 tests (to be written): manifest v5 refuses real work when the chain tip is below sequence 2 and checks
  the chain receipt per unit; `phase3._bind_view` refuses a unit without its fold's amendment-2 dry-run evidence;
  `phase4.replay_cpu_logits`.
- All use the same synthetic contract fixture; no side copies its expected values into its own tests.

## Hand-off order

1. Science: replay goldens and the contract fixture with `amendment_rules.py`, then `amendment_chain.py` against the
   fixture with its synthetic tests (staged).
2. Execution v5: `runtime_precision.py`, the comparator CUDA path, the delegated `SCORER_CODE_PATHS` line, manifest v5
   (which checks the chain receipt with `check_chain_receipt`).
3. Science: the remaining scientific packages, rebased on step 2.
4. Execution v5: plan-v2 dispatch, futility gate, CLEAN sanity hooks, F6 input build, refit and fold 1-4 queues,
   G2 v2.
5. Protected-source freeze, then the five plan-v2 records, the five dry-run-v2 runs, the required role files and the
   record build (`amend-2` with `--plan-v2 FOLD=PATH`, `--dry-run-receipt FOLD=PATH` and `--bind ROLE=PATH`); the
   dry-run-v2 receipts, the tie exposure and the F6 inputs are written once below
   `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/amendment2/` and bound privately.
6. The maintainer reviews and pushes the link, its rendering and its log line in one commit;
   `verify-amendment-chain` runs in the workstation's clean full checkout at the published `origin/main` HEAD, which
   holds the private files at their bound paths below
   `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/amendment2/`, and writes the receipt that the pods check
   (a fresh clone of `origin/main` holds no private file and is refused by C8 and C11).
