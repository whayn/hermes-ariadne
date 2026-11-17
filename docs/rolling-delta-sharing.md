# Completed delta source sharing: bounded follow-up to #19

Baseline: `b7b432a54166c126def78e6c22a54def131539d2` (#39).
Measured runtime/tests: `2dc33570cb6c810ff94125055757b5e3235c078a`.

**The <=5 s cold-and-warm usability gate still fails.** Native 100,000-event
cold open was 107.999 s in the single-run candidate control. This is a bounded
CPU-work improvement, not first-open usability, production readiness, activation
approval, provider latency or model-recall evidence.

## Scope and acceptance boundary

Schema-3 generation validation already canonicalizes source rows, verifies their
ordered identities and prefix extension, renders the readable projection and
computes exact summary-input bytes. This slice stages those bytes only after the
whole generation passes. The journal makes the evidence eligible only after its
existing ready/published job, validated text and frozen-target cross-check passes.

A frozen target still passes its own scope, request/route, parent and size admission.
Its exact payload is reread/hash-bound as before. A source-proof hit requires the
same open's memo/store/session/settings/policy/index, the exact accepted generation
and its validated prefix state, the same operation-bound parent and identical
parent-state object, matching ordered delta identities/ranges/frontier/prefix/job,
and byte-for-byte readable-input equality. A root has an explicit same-memo empty
parent case. Capture checks and final target serialization remain. Incompatible
or unavailable evidence selects the existing full validator, not a new refusal.

The completed generation's extension verdict is reused only against that identical
validated lineage and delta. This is not a generic row cache, a dispatch permission,
or a replacement for journal accounting/recovery checks. Every generation artifact
is still enumerated; every new open still verifies all referenced archive sources.
Target-source validation does not attest the target's own generated output file:
neither the old full checker nor this shortcut reads that file just to validate
the unchanged frozen source target. Loading the generated view, binding it as a
descendant's parent, and startup enumeration retain their existing fresh artifact
checks. Completed source evidence does not make a changed output artifact usable.
The [public-boundary full/optimized reproduction](https://github.com/whayn/hermes-ariadne/pull/40#discussion_r4601164896)
confirms both accept the unchanged frozen target, deny another reservation for its
published job, and refuse the changed generation on load or a new open with the
same exception. Its actual output is also in the machine evidence.
Supported source writes remain insert-only. Direct SQL mutation within an open
remains outside the existing snapshot contract; the next open detects it.

No serializer, schema/artifact version, settings digest, durable identity, source
value/coverage, frozen request/route, privacy/restriction/carrier policy, reservation,
logical facade-job charge, recovery or migration behavior changes. No proof enters
the cross-open cache. Raw generation validators do not create this collector.

## Bounds and lifetime

Staged plus accepted evidence has at most **16,777,216 expected-input bytes** and
**256 fixed-shape records**. This bounds the new retained input payload/count,
not total RSS or the existing verification cache. Pressure discards a candidate
and falls back. Candidates belong to one walk and are released on every exit;
accepted evidence belongs to the open's memo and is discarded with it on close
or failed operation. Parent artifact binding remains operation-local and fresh.

After exact comparison, accepted evidence borrows the immutable readable bytes
already owned by its parsed target, rather than retaining a duplicate; the byte
allowance conservatively counts their whole length anyway. Views, delta segments
and parent states are borrowed, with no full-prefix copies or new per-generation
identity sets when `cache=None`. With the existing cache enabled, target promotion
uses the already-retained generation delta identity sets with unchanged values,
keys, bounds and freshness policy.

This is deliberately **generation -> target only**. Without a pending successor,
the last head target precedes its own generation check and falls back. Do not
change audit/refusal order or claim universal once-per-delta execution.

## Deterministic RED/GREEN and preservation

The eight new cost cases failed on the previous code. Real reader/checker wrappers
then measured the following two-row-delta fixtures with a ready pending successor,
both with no verification cache and with an initially empty one:

| Published generations | Canonical row visits before | After | Source reads | Startup artifact reads | Reconciliation artifact bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 34 | 18 | 18 | 8 | 17 |
| 16 | 66 | 34 | 34 | 16 | 33 |

Every distinct row is canonicalized once in that pending-successor fixture. A
fully published G-generation chain without pending instead takes G+1 calls for
G source deltas, because the head delta is checked twice. For the 64-generation
benchmark, calls to canonicalization, rendering and summary-input construction
are each **128 -> 65**. The benchmark's warm path makes no canonicalization calls.

The frozen b7 full validators in `tests/rolling_delta_reference.py` independently
replay the replaced paths. Tests compare exact values and exception type/message
at the same restored pathname under no, cold and warm cache, including next-open
source/session corruption, altered/missing artifacts, manifest/target hashes,
rewritten payload plus stored hash, job text/state and charging corruption. Other
cases cover wrong scope/memo/index/view/parent state/input, budget pressure,
borrowing/lifetime, capture interruption, and late failure after new evidence was
accepted but before startup completed. All existing artifact/source bounds and
literal legacy/delta identity pins remain.

Affected validation/storage/identity/recovery/migration and benchmark suites:
**337 passed in 115.37 s**. Ruff check/format and diff checks passed. Eight isolated
mutants were killed: acceptance, exact input, parent state, scope, record bound,
byte bound, capture and pending cleanup. Mutant imports were verified to use their
isolated source copies; the real runtime source remained unchanged.

### Final frozen integration and packaging

The canonical pinned-host suite on `69aae948eff478a83141f7032781b0282e661ffe`
(the evidence-doc commit, with runtime/tests unchanged from `2dc3357`) returned
**3,830 passed, 2 skipped in 908.84 s**. The skips were HTTPX's optional zstandard
coverage (dependency absent) and the opt-in native tracer. The tracer was then
exercised separately from a wheel installed outside the checkout:

- Native schema-3 installed restart: **1 passed in 27.01 s**.
- Actual old-wheel publication -> explicit offline migration -> newly installed
  plugin -> fresh-process restart: **1 passed in 27.00 s**.

Both routes use the pinned Hermes source and disposable OpenAI 2.24.0 / HTTPX
0.28.1 SDK environment, exercise two automatic advances and restart, and verify
canonical originals/accounting through the independent native tracer. They are
synthetic scripted-provider mechanics, not real-model recall evidence. The
optional zstandard branch remains unexercised.

Wheel and sdist builds passed. Byte audits matched all **26 source Python modules**
against the wheel, sdist and installed package, all **19 runtime files** in each
native/migrated installed profile, and LICENSE/NOTICE against frozen source.
The first scratch audit runner failed because it assumed the runtime allowlist
was an AST literal; after handling its inspected static dict comprehension, the
audit passed without production changes. That runner failure is not counted as
a successful audit or a failed plugin test.

Hosted CI run [44104073400](https://github.com/whayn/hermes-ariadne/actions/runs/44104073400)
passed on the same frozen head: Quality, Secrets, Hermes integration and archive
Python 3.11/3.14. PR conventions were skipped on manual stacked dispatch. Later
review-clarification documentation does not change the measured/tested runtime;
the tested SHA above remains explicit.

## Sequential checkpoint campaign

Python 3.14.9 on Linux aarch64, inherited CPU affinity recorded at launch. The
machine-readable [evidence](rolling-delta-sharing-2dc3357.json) records the actual
CPU information, affinity and source heads. No taskset/governor changes or
exclusive-host claim. Local timing ran sequentially without competing local test,
build or profile jobs; hosted CI ran remotely.

**24 completed cold/warm pairs, 12 groups, six unchanged fixture inventories.**
Native 256/4,096/16,384 groups have three repetitions per variant; native/migrated
100,000 and legacy 256 controls have one pair per variant. Cold means an empty
verification cache, not a flushed OS page cache. Both variants use the same new
counter-instrumented benchmark harness with their own frozen runtime on PYTHONPATH.
Do not mix its samples with older campaigns' different harness measurements.

Low-RSS fingerprint children verify before/after active SQL/artifact commitments,
exact archive hashes, root inventories/metadata and retained migration snapshots.
All before/after values and cross-variant/repetition inputs matched. The evidence
stores each of the six detailed inventories once and each run's matching SHA-256
commitments; completeness, group counts and input equality were checked in code.
Every cold and warm source-check count equals the fixture's event count.

### Three-repetition native checkpoints

Seconds; cold cells are median [minimum–maximum], warm cells are medians.

**Fixed-order limitation:** every repetition and control ran baseline before
sharing. Identical fixtures and sequential execution do not control OS-cache,
thermal or clock-frequency order effects. These samples do not establish whether
that order biased the timings or in which direction. A future comparison should
alternate or randomize variant order and record it; the historical launcher and
observed timings below are unchanged.

| Events | Baseline cold | Sharing cold | Baseline warm | Sharing warm |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.605 [0.597–0.651] | 0.534 [0.529–0.542] | 0.153 | 0.159 |
| 4,096 | 5.810 [5.785–5.830] | 4.682 [4.550–4.701] | 0.228 | 0.228 |
| 16,384 | 22.482 [22.129–23.165] | 17.649 [17.430–18.150] | 0.511 | 0.480 |

### Single-run controls — not medians or statistical speedup evidence

| Storage / events | Baseline cold | Sharing cold | Baseline warm | Sharing warm |
| --- | ---: | ---: | ---: | ---: |
| Native / 100,000 | 139.837 | 107.999 | 2.330 | 2.295 |
| Migrated / 100,000 | 143.274 | 107.438 | 2.405 | 2.244 |
| Legacy / 256 | 0.696 | 0.728 | 0.196 | 0.197 |

These are observations, not a repeated cap speedup claim. The earlier storage
matrix and ancestor-binding evidence retain their own source revisions and limits.
Legacy canonical/render/input/encode counters and artifact reads are unchanged;
its one-shot time difference is not evidence of a code-level regression or gain.

### Work and memory

For native and migrated cap controls:

- Canonical row visits: **200,000 -> 101,560**. The unmatched head delta accounts
  for the remaining duplicate visits; the ordinary raw generation source-prehash
  loop is intentionally unchanged.
- Canonicalization/render/input-builder calls: **128 -> 65** each.
- Artifact reads stay **191 cold / 128 warm**; all **100,000 source checks** stay
  present on cold and warm opens. Stored target/artifact bytes and identities stay
  unchanged.
- Proofs at the end of candidate startup: 64 accepted records, 63 hits, one miss,
  zero pressure; 15,071,058 bytes counted against the 16 MiB allowance. Warm opens
  retain no new proofs, because their existing known-generation path has no newly
  computed exact input to stage.
- Encoded JSON output: native **415.37 -> 300.75 MiB**; migrated
  **415.44 -> 300.82 MiB**. Encode counts: native **301,846 -> 203,281**;
  migrated **301,857 -> 203,292**. These count produced bytes, not HTTP bodies,
  archive I/O traffic or provider tokens.
- Single-run peak RSS: native **418.5 -> 402.5 MiB**, migrated
  **418.7 -> 396.5 MiB**. These samples do not establish a general memory reduction.
- Retained verification-cache object estimate: **103.8 -> 78.1 MiB**. Identical
  target/generation delta identity sets are now shared instead of rebuilt into
  duplicate objects; cache key/value semantics and entry/source-id counts are
  unchanged. This is a reachable-object estimate, not RSS, not a hard cache byte
  limit, and not proof-payload allocation peak. No tracemalloc claim is made.

At 16,384 events, peak RSS medians were 94.2 -> 91.8 MiB and retained-cache
estimates 16.5 -> 12.6 MiB. These are separate metrics, not interchangeable limits.

## Synthetic public callback controls

Fresh 32-generation fixtures exercise actual select/navigation/completion callbacks
against the existing injected DB/facade. Each trace has a cloned pre-completion
fixture so completion is not an already-completed shortcut. Both sources end at
33 publications. These single samples use freshly seeded synthetic fixtures, not
the six shared cap/checkpoint fixtures, and are not provider/model-recall evidence.

| Callback | b7 unprofiled seconds | Sharing unprofiled seconds |
| --- | ---: | ---: |
| Select | 0.531845 | 0.484342 |
| Navigation | 0.344177 | 0.298867 |
| Completion | 1.045506 | 0.938259 |

Instrumented profiles and unprofiled timings are distinct; overlapping cumulative
profile times must not be summed into a speedup estimate. Separate fresh no-cache
16,384-event controls took 22.406 -> 17.320 s unprofiled; their subsequent cold
profiles took 55.841 -> 38.469 s instrumented and confirmed 128 -> 65 canonical
calls. These additional controls are not folded into the three-repetition table.

In the candidate cold profile, canonical JSON encoding still accounts for
21.282 s cumulative, `_canonical_rows` 11.421 s and EventStore source reads
10.278 s; these overlap. Startup record validation remains 36.728 s cumulative,
while reconciliation falls from 18.570 to 1.706 s. Removing duplicate target work
does not remove the remaining generation/source-validation floor. This is a
diagnosis for a separately scoped follow-up, not permission to change serializers
or archive access here.

The [reproduction archive](rolling-delta-sharing-reproduction.md) preserves the
exact executed campaign, profile, callback and mutant launchers, their hashes,
actual launch command and historical PID barrier. Machine evidence includes the
fresh cold profile text and callback results. Fixture commitments were unchanged
through the additional cold profiles as well as the main campaign. The actual
runtime CI was manually dispatched because this PR is stacked; its PR-conventions
job is skipped, not certified by that dispatch.

## Limits, review and rollback

The cap first-open gate remains failed; #19/#26 remain open. No real-provider
latency, summary quality/recall, prompt-injection resistance or daily readiness is
established. Serializer optimization, archive batching, source-prehash sharing,
bidirectional reuse and broader cross-open caching remain separately scoped work.

Reverting the optimization restores full duplicate validation without migration.
It affects future validation, not outputs already generated or actions already
taken. Source-proof provenance is an integrity-sensitive boundary: automated tests
and bot reviews are not human approval. No merge, release or live activation is
authorized by these measurements.
