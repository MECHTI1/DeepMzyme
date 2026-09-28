# GVP5 input and numerical diagnostic, fixed before execution

Reuse only_gvp__five_class__none__fold0__seed42 and its original checkpoint,
saved normalization, validation UID order, batch size 16 and FP32 settings.
No fitting, checkpoint selection or held-out access. Original failed replay and
the original absolute 1e-6 gate remain untouched.

Run the existing unchanged audit_pmm_replay.py against the frozen source.
Independently reconstruct every validation graph with raw cache disabled and
compare its full tensors to checksummed training-window cache contents in
namespace 7e9402a3ec2149bb4a722fe58999ae4d6309c87b4a36f6986184ee7b01b2785b,
between 2026-09-28T03:32:30Z and 2026-09-28T04:13:33Z. These boundaries come
from the preserved training-child start and exit events; replay began later.

If the original input audit fails solely on the previously documented disabled
EC target (cached y_ec=0 versus reconstructed -1), retain the failure and use
the existing v1.1 recover operation. It must reconstruct the independently
captured descriptor bytes exactly, check all collated predictive tensors and
prove unchanged CPU metal logits/loss for the first complete batch. Any other
mismatch stops diagnosis. No metal target or model input may be ignored.

Two fresh processes per condition, five full passes per process: original FP32
and strict deterministic evaluation with warn_only=False (20 passes total).
Preserve all passes and unsupported-operation failures. Check unchanged inputs
and model state, native-five and probability-collapsed-four outputs, and every
preserved original/replay export. No repeat-until-pass or probability averaging.

The separate core engineering contract uses absolute 1e-5/relative zero plus
exact class predictions and reconciled class metrics. This is an explicit
retrospective qualification for this historical fit, conditional on complete
diagnosis; it never rewrites its strict failure or establishes determinism.
Future-fold policy is bound before training. A diagnostic summary alone does
not certify the full core grid, promote a model or authorize held-out use.

One owned L4 session within unchanged session/day caps. Diagnosis and the
separate bounded concurrency probe use no more than a requested one-hour
allocation (57-minute provider STOP), with a 15-minute closeout reserve.
Preserve verified host copies before stopping. No automatic replacement/restart.
