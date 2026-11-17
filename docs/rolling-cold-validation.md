# Cold delta ancestor binding: bounded follow-up to #19

Baseline: `eeb5e0c8b3b8ab37ff5769151efb2d17be7c6fa6`.
Measured implementation: `b3a8d1dab1f18607ee84383e0180a7ef809c83f7`.

**The <=5 s cold-open gate still fails.** This change removes repeated ancestor
reads. It is not a cap-sized latency solution, an activation approval, or a claim
that source canonicalization now runs only once.

## Scope and invariants

Schema-3 startup record validation now owns an operation-local binding scope,
without publishing cache facts before startup reconciliation succeeds. Explicit
delta-parent lookups reuse a parent only when it is already hash-bound in that
operation, journal-accepted, tied to its validated prefix state, and in the same
store/session/policy/settings scope. Hits retain capture checks. With no binding,
or a wrong-name artifact, validation follows the original exact-byte/refusal path.

The next operation rebinds ancestors. Every artifact remains enumerated, including
interrupted-publication artifacts. Source verification, machine restrictions,
identity and exact-readable-input checks, reservations, recovery and SQL job
cross-checks remain. Schema2 follows its previous startup path. There is no stored
format/settings/identity change, added memo collection, widened cross-open cache,
serializer replacement, canonical-delta sharing or archive batching.

## Deterministic RED/GREEN

Real-reader wrappers on native delta chains with a ready pending successor:

| Published generations | Startup record reads before | After | Reconciliation reads after | Source rows read | Rows canonicalized |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 36 | 8 | 17 | 18 | 34 |
| 16 | 136 | 16 | 33 | 34 | 66 |

These results hold with `cache=None` and with an initially empty verification
cache. The cost regression fails on the baseline and passes on the implementation.
Canonicalization counts intentionally do not change in this slice.

Startup record audit reads each parent once. Reconciliation has its own binding
scope, enumerates every artifact and reads the manifest head: at most `2G + 1`
reads for the pending-successor fixture, independently of the first audit's `G`
reads. A public operation starts a new scope; bindings are not journal-open-wide.

## Sequential checkpoint campaign

Linux aarch64, two CPU-affinity slots, Python3.14.9. Existing committed
`scripts/bench_rolling_open.py`, synthetic native-shaped Codex exchanges,64generations.
No model/provider calls. **24 completed cold/warm pairs across12groups**:
three repetitions per native256/4096/16384 group and one pair per cap/migrated/legacy
control, each on baseline and implementation.

Cold means an empty verification cache, **not cold OS page cache**. Low-RSS
launcher; independent fingerprint subprocesses run before and after each pair.
Measurements are sequential, without competing local tests/builds/profilers.
Exact active SQL/artifact fingerprints, archive hashes, root artifact/metadata
hashes and retained-snapshot hashes remained unchanged on all24pairs. Both cold
and warm source-check counts equal the fixture event count.

### Three-repetition native checkpoints

Times are seconds; cold cells show median [minimum–maximum]. Warm values are medians.

| Events | Baseline cold | Binding cold | Baseline warm | Binding warm |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.806 [1.750–1.899] | 0.592 [0.574–0.616] | 0.182 | 0.153 |
| 4,096 | 6.979 [6.914–6.989] | 5.765 [5.656–5.968] | 0.254 | 0.232 |
| 16,384 | 23.466 [23.350–23.766] | 22.503 [22.370–22.717] | 0.515 | 0.507 |

### Single-run controls — not medians or statistical speedup evidence

| Storage / events | Baseline cold | Binding cold | Baseline warm | Binding warm |
| --- | ---: | ---: | ---: | ---: |
| Native /100,000 | 141.406 | 140.416 | 2.378 | 2.221 |
| Migrated /100,000 | 142.135 | 137.609 | 2.430 | 2.279 |
| Legacy /256 | 0.677 | 0.664 | 0.188 | 0.174 |

Do not substitute these cap single runs for the earlier three-run matrix in
[delta-storage evidence](rolling-storage-benchmark.md). In particular the small
native-cap time difference does not establish a meaningful statistical speedup.

Across every native/migrated checkpoint, cold artifact reads fell **2,207→191**;
warm reads fell **191→128**. Legacy reads stayed191. At the cap, both new-format
variants still verify100,000sources per cold/warm open and store200,000event-ID
references. Serialized payload counts are unchanged.

Cap retained-cache estimate is unchanged at103.8MiB. Clean-launch cap RSS samples
were418.7→418.1MiB native and419.3→418.6MiB migrated; these single observations do
not establish a memory reduction. RSS, retained-object estimates and allocation
peaks are different metrics. No tracemalloc allocation-peak claim is made.

## Callback/profile evidence and remaining floor

A fresh baseline16,384-event cold profile found2,207generation artifact reads and
128canonical-delta checks. The untraced open was23.569s; the separate instrumented
open was57.448s. Instrumented cumulative times are not wall-time predictions:
`_validate_records`39.235s, strict encoding34.523s and `_canonical_rows`21.970s
(overlapping call stacks, not additive costs).

A separate scratch driver used the existing `test_rolling_engine.py` injected
DB/Facade fixtures to build32generations, then measured public select, navigation
and completion with a fresh verification cache per callback. Completion published
G33. Profiles used a separate cloned synthetic home; the table contains untraced
single runs, not profile timings or provider latency:

| Callback | Baseline seconds | Binding seconds |
| --- | ---: | ---: |
| select_context | 0.851 | 0.517 |
| navigation | 0.634 | 0.327 |
| on_turn_complete | 1.243 | 0.977 |

These pre-commit implementation samples preceded only a comment edit. They are
not cap-sized callback measurements, real-model recall, summary quality, gateway
load or native provider end-to-end evidence. The complete installed-Hermes
loopback restart checks are a separate verification tier reported in the PR.

## Verification and stopping point

New tests compare the replaced eeb5e0c parent-resolution/startup paths with the
optimized paths on valid/corrupt fixtures, including exact exception messages.
They restore the same synthetic pathname between runs so filesystem errors also
compare exactly. Tests cover next-operation distant-ancestor tampering,
post-failure pending-target coverage, capture/reentry, cache non-promotion after
failed startup, scope mismatches and accepted-prefix object ties. Existing legacy
and delta identity pins remain unchanged.

Five scratch guard mutants were killed with real assertion failures: stale epoch,
missing scope checks, dropped capture checks, premature promotion and untied prefix
reuse. Full canonical/installed-package/CI evidence belongs to the PR and must be
read at its reported revision; these performance tables are not a substitute.

The approved slice stops here. Repeated source serialization and generation/target
canonicalization remain; point-read overhead also remains. #19/#26 stay open.
Sharing completed canonical-delta facts is a **separate next approval**, not an
implicit part of this result. Do not weaken integrity or expand scope to make the
latency target appear met.

[Machine-readable checkpoint results](rolling-cold-validation-b3a8d1d.json) include
all24pairs, medians/ranges, source/artifact/encoding/memory counters, and before/after
inventory commitments. To reproduce, use separate frozen baseline/implementation
worktrees, the committed benchmark `measure HOME --in-place --cache`, identical
synthetic fixtures, fresh processes, and separate fingerprint processes; never
run timing alongside tests/builds or migrate a baseline original in place.
