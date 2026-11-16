# Delta-only rolling storage contract (#26)

Status: approved design, implementation in progress. This document specifies
storage, not provider compatibility, summary accuracy, activation or release.

## Versions and immutable meaning

Legacy journal schema 2 uses artifact format 1 and untagged whole-prefix targets.
Its codec, identity pins, settings digest, prompt, policy and projection remain
unchanged. New journal schema 3 uses artifact/target format 2, never mixed chains.
Legacy sessions stay legacy until an explicit offline conversion. Schema 1 is
still refused: absent actual charging identities cannot be reconstructed.

New target keys: `format`, `version`, `session_id`, `settings_digest`,
`parent_generation`, `parent_summary_sha256`, `delta_event_ids`,
`delta_message_uids`, `delta_start`, `delta_stop`, `frontier_count`,
`prefix_sha256`, `readable_input`, `request_bytes`, `route_labels`.
New generation keys: the same format/session/settings/parent/coverage keys plus
`policy_id`, `readable_projection_id`, `prompt_version`, `sequence`,
`summary_text`, `summary_sha256`, `summary_input_sha256`, `job_id`, and without
`readable_input`, `request_bytes`, `route_labels`.

All key sets are exact, canonical ASCII JSON, duplicate members refused, integers
not booleans, bounded before untrusted allocation. The delta is nonempty;
`len(delta_event_ids) == len(delta_message_uids) == delta_stop - delta_start`;
`delta_stop == frontier_count <= max_prefix_events`. Roots start at zero with null
parents and sequence 1. Successors bind the exact validated parent generation and
summary digest and start at its frontier, sequence incrementing by one. Source
hashes, UIDs, session, complete exchanges and identity collisions are validated,
including collisions with distant ancestors; siblings do not collide unless they
share lineage.

Targets and artifacts hash their NEW exact bytes. Job derivation stays the legacy
algorithm applied to the NEW target ID. Old IDs never refer to modified bytes.
`prefix_sha256` remains SHA256 of the exact legacy canonical JSON array of ALL
ordered event IDs. It is not a Merkle root or hash chain. A per-open streaming
hash state can extend the array before its closing bracket; no hash internals are
persisted. Summary coverage echoes and exact readable/frozen facade input bytes
stay unchanged by conversion. Those bytes are not provider HTTP-body evidence.

Validated coverage uses immutable parent/delta segments. Do not retain flattened
prefix arrays per ancestor/cache entry. Cache scope includes storage encoding;
source/artifact bytes, lineage, accounting and recovery remain freshly checked.
Identity-reference storage is linear in distinct chain events plus generations,
with separate bounded per-job request/summary/input storage and pending/orphan
coverage. Fresh source checks remain linear; no constant-time request claim.

## Stable root, selector and migration files

Let `R` be `ariadne/rolling/<sha256(session_id)>`. `R/writer.lock` is the same
cooperative exclusive lock for old/new readers and migration; never rename R or
replace its lock inode. Acquire it before choosing the active data path.
All files/directories must satisfy existing private/no-symlink/single-link and
SQLite-companion checks before archive reads or destination writes. All sibling
paths are fixed names, never strings supplied by a selector.

Proposed layout:
- legacy: `R/journal.sqlite3`, `R/generations/`, `R/manifest.json`;
- new: `R/v3/journal.sqlite3`, `R/v3/generations/`, `R/v3/manifest.json`;
- snapshot: `R/legacy-snapshot/` exact legacy DB/artifact/manifest bytes;
- permanent receipt: `R/migration.json`;
- durable migration intent: `R/migration-intent.json`;
- active selector: `R/storage.json`.

Selector exact keys: `format: "ariadne-rolling-storage"`, `version: 1`,
`schema: 3`, `receipt_sha256` (digest of canonical migration receipt),
`session_id`, `settings_digest`. Native-new journals use a separate explicit
native selector with `receipt_sha256: null`; the root schema fence must exist
before publication. A missing selector alongside v3, migration, snapshot or fence
state is NOT an empty session. Ordinary runtime opens refuse incomplete upgrades;
only the explicit offline migrator may complete them.

Receipt exact keys: `format: "ariadne-rolling-migration"`, `version: 1`,
`session_id`, `settings_digest`, `source_fingerprint`, `destination_fingerprint`,
`targets`, `jobs`, `generations`. Mappings are old-ID→new-ID objects, permanent
private audit data, never prompt material. Fingerprints hash a canonical mapping
of relative artifact paths to exact byte hashes and canonical logical SQL rows
(including blob hex), excluding volatile SQLite page-layout and lock contents.
Receipt also binds exact legacy snapshot file hashes; source evidence must not
be confused with a post-recovery snapshot.

Intent exact keys: `format: "ariadne-rolling-migration-intent"`, `version: 1`,
`session_id`, `settings_digest`, `receipt_sha256`, `operation` (`upgrade` or
`rollback`). A single immutable intent identifies the audited operation; phases
are determined from exact validated disk state, not a mutable unchecked flag.

## Offline conversion ordering and crash states

No facade invocation, new reservation, counter reset, prompt regeneration or
source rewrite is permitted. Stop session runtimes before conversion; the lock
excludes old readers too. Do not use normal `RollingJournal` construction as a
source validator: it marks reservations uncertain and can publish a ready result.
Use an admitted read-only legacy audit and record exact original evidence first.

1. Admit and lock; read-only validate ALL source records/artifacts and campaign,
including valid generation-only/manifest-only interrupted publication. Tampered
input refuses before staging. Snapshot exact source, including SQLite companion
handling under exclusive ownership; no unsafe links or foreign files accepted.
2. Build v3 topologically: parent generation → target → job → generation, including
planned/failed/uncertain targets and ready artifacts. Remap every SQL reference,
manifest, filename and publication pointer. Preserve every non-derived byte/field
and charge. Reserved stays reserved until normal new-format startup marks it
uncertain; migration itself grants no retry permission.
3. Read-only validate staged destination and field-by-field equivalence. Sync DB,
artifacts, snapshot and containing directories. Write receipt and fsync, then
write/fsync intent binding it. Pre-intent orphan staging is refused by ordinary
opens; explicit retry may reuse ONLY an exact audited stage, never assume it valid.
4. After exact backup/receipt/intent are safe, set the legacy root DB's
`user_version` to a dedicated unsupported upgrade fence. Sync it. Old code must
refuse instead of ignoring the selector and reopening a stale spend ledger.
5. Atomically write/fsync `storage.json`; new opens inspect intent/selector/fence
BEFORE schema admission, initialization or reconciliation. Selected state requires
exact selector/receipt binding, schema-3 data and legacy fence. Never initialize a
new campaign because one of these files is absent or contradictory.
6. Reopen in a separate process; clear scope caches; validate/reconcile using
existing semantics. Repeated explicit migration returns the same mapping; it
never dispatches a job or creates another campaign.

Crash states:
- no stage/intent/selector: ordinary legacy semantics;
- stage/snapshot/receipt but no intent: ordinary runtime refuses; explicit audit
can resume or refuse without touching original charges;
- intent + source still schema2: explicit retry revalidates both snapshots before
fencing/selecting; ordinary runtime refuses;
- intent + fence + no selector: explicit retry selects ONLY exact validated stage;
ordinary runtime refuses (no missing-active-path initialization);
- selector + fence + valid receipt/v3: ordinary new-format recovery;
- any malformed, missing, conflicting or hash-mismatched required artifact: refuse.

Tests must process-kill at each fsync/commit/rename boundary, test old/concurrent
writers and changed sources, and prove that removing each accounting/fence guard
is detected. Two directory renames are never described as one atomic switch.

## Rollback

Keep snapshot/mapping after successful opens. Simple rollback requires destination
state exactly migration-equivalent, including all lifecycle fields—not merely no
new generation. A reserved→uncertain or ready→published recovery is divergence.
Before any such mutation, under the same lock, persist a rollback intent, restore
and sync exact legacy artifacts/DB, remove/sync selector, then complete a durable
rollback receipt transition. Runtime refuses interrupted rollback. After any new
charge or lifecycle divergence refuse; audited reverse migration is separate
scope. Deleting a journal or reinstalling old code is never rollback.

## Acceptance evidence

TDD codec/journal/migration tests, unchanged legacy pins, warm/uncached mutation
verdicts, all lifecycle preservation and process-death recovery. Benchmark native
and migrated v3 versus legacy using `scripts/bench_rolling_open.py`, including
100,000 events / 64 generations and fixed-events depth scaling. Proposed/approved
gate: cold AND warm median of three sequential untraced opens <=5 s on the same
host, plus separate full select/navigation/completion timings, bytes/read counts,
RSS/retained memory and migration peak disk/time. The existing benchmark raises
byte limits and is synthetic, not default-settings chat latency.

Run canonical full tests/builds and outside-checkout wheel/sdist proof on stable
files, installed-host loopback with two advances and restart. Report misses/skips;
keep #19 open if first-open or end-to-end usability is not established. No live
activation, real-provider campaign, release or merge is included.
