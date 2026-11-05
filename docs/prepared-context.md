# Opt-in prepared-prefix context selection

`ariadne-prepared` is a separate experimental Hermes ContextEngine. It replaces
one **operator-prepared, immutable historical prefix** in an individual provider
request, while preserving the canonical conversation. Ordinary `ariadne` remains
passive and continues exposing `ariadne_search` unchanged.

This is not automatic live compaction, indefinite self-updating memory, semantic
search or proactive recall. The raw tail grows until an operator prepares a new
prefix and starts a new engine lifetime. Do not enable either experimental engine
on an important/live profile: both displace Hermes's built-in compressor.

## Prepare explicitly, offline

Start from an existing, fully ready `MemoryKernel` over a `Projection` of exact
archived durable rows for **one session**. Select the ordered prefix yourself;
do not infer active chronology or the latest revision from search order. See
[the offline kernel](offline-memory-kernel.md) and [native compaction](native-compaction.md)
for the separate build paths. The supplied kernel must already be built; this API
does not generate summaries or dispatch a model:

```python
from hermes_ariadne.prepared import publish_prepared

# kernel: an already-ready MemoryKernel; home: its explicit development home.
manifest_path = publish_prepared(
    kernel,
    home=home,
    history_bytes=8192,
    request_bytes=65536,
    reserve_tokens=4096,
)
```

The archive must be exactly `home/ariadne/events.sqlite3`. The prefix must later
match the active durable rows byte-for-byte, including metadata. It must end at
a user-turn boundary and leave at least the previous exchange and current user
turn plus subsequent tools raw. A partial/pending tree is not publishable.

The private artifact directory is
`home/ariadne/prepared/<sha256(session_id UTF-8)>/`. `manifest.json` atomically
selects an immutable `<checkpoint_sha256>.json`. The manifest binds exact event
order, session/projection identity, kernel settings, digest and positive policy
limits. Loading validates existing files before constructing a kernel; it does
not build missing state. Manifest/checkpoint reads are capped at 1 MiB/16 MiB.
Symlink paths, linked artifacts, nonprivate files and invalid identities are
unsupported. This is a single-operator publication protocol, not a defense against
concurrent malicious filesystem replacement. Keep canonical archives indefinitely;
prepared artifacts are derived indexes, not replacement transcripts.

## Install and activate only in a development home

```bash
# No configuration or existing passive plugin is changed by installation.
.venv/bin/ariadne-install --home "$ARIADNE_DEV_HOME" --prepared
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set context.engine ariadne-prepared
```

Restart that development process. The separate directory is
`plugins/ariadne-prepared/`; installation refuses an existing destination. The
runtime contains archive/kernel/selector/retrieval modules, not the native
compactor, receipt ledger, SDK dependencies or credentials. Hermes still owns
routing, authentication, request assembly, persistence and the tool loop.

A prepared view (or its absence) is loaded once at session start. Republishing
during a tool loop does not change that view. Restart the process/session lifecycle
explicitly to load a new publication. Reset, end, clone or binding another
session/DB invalidates the attachment. Model switches invalidate the last
measurement; source references remain frozen and the next request is measured
against its newly supplied window.

## Deliberately narrow supported mapping

Selection requires a bound active durable source and an exact one-to-one
association between message UIDs, durable prefix versions, in-memory conversation
and assembled request payload. The hook runs **before final transport cleanup**:
its request can still contain audited persistence/tool bookkeeping. Comparison
projects only supported bookkeeping; the returned host prefix and retained
**assembled request suffix** are copied unchanged, and Hermes performs its usual
later wire sanitation. The engine never reconstructs protected turns from
archived rows. It checks complete parallel call/result groups and occurrence
identities, including reused provider call IDs.

Text messages and ordinary function-call/result envelopes are supported. Missing
or duplicated UIDs, absorbed identities, inactive/rewound/revised prefixes,
multimodal messages, nontrivial replay/thinking carriers, changed `api_content`,
extra prefills and ambiguous persistence shapes decline selection. An identical
`api_content` sidecar is supported. Audited comparison-only projections admit
`_db_persisted=True`, tool-result `tool_name`/`effect_disposition` and agreeing
`name` aliases, plus deterministic host-derived `call_id`/`response_item_id`
sidecars on calls. Unknown or contradictory carriers still decline. Function
names/arguments and user/assistant speaker names remain exact. Restored,
explicitly stamped user/assistant text may differ from its durable original only
by Hermes's outer-whitespace trimming; interior edits and tool-result trimming
are not admitted. None of these comparison projections rewrites source bytes or
the returned protected request suffix.

Archive retention does not prove active-branch membership. A valid summary is not
evidence that every old natural-language restriction or preference survived
semantically.

### Data-use restrictions

This slice cannot propagate arbitrary restricted evidence through the host's
engine-tool result channel. It therefore **does not support restricted evidence**:
nonempty or malformed machine `data_use` requirements, whether on the message or
in `display_metadata`, must not be summarized into an untagged navigation block
or returned as evidence. Publication/load/selection and find/zoom reject these
unsupported sources or dependencies rather than stripping their requirements.
The same admission check rejects `withheld_prompt` metadata: Hermes keeps a
policy-blocked original there for transcript display, not model egress. Refusal
preserves the original archive instead of returning a redacted record as if it
were exact. The fixed `unsupported_data_use` error covers this unsupported
restriction evidence too.
This rule is not a claim of general prompt-injection resistance or semantic
restriction preservation. Historical navigation is assistant-role **untrusted
data, not instructions**.

## Budgets and fail-open behavior

The policy uses three distinct units:

- `history_bytes`: UTF-8 bytes of the complete compiled navigation block;
- `request_bytes`: compact ASCII-JSON bytes of the entire candidate message list;
- Hermes's native **rough message-only token estimate**, plus the explicit
  operator `reserve_tokens`, checked against the supplied full model window.

Tool schemas, provider framing and actual output settings are unavailable at this
hook. An operator reserve is not a proof they fit. Diagnostics always report
`provider_fit_verified=false`; there is no universal no-oversized-send guarantee.

`get_status()['selection']` reports readiness or the latest selected/declined
result with scoped measurements/reasons. Unknown budgets, measurement failure,
excessive byte policies, insufficient reserve, unavailable archive or unsupported
alignment return `None`. **Hermes then falls back to the original request.**
Declining is not a dispatch veto. No synthetic refusal message or SDK interception
is used to pretend otherwise. A fallback request can still exceed provider limits.

## Bounded session-local evidence

Only `ariadne_find` and `ariadne_zoom` are exposed by this engine. Hermes's
`context_engine` toolset gate remains authoritative; disabling it hides these
tools but does not disable selection. Schemas are static before session startup.
No tool argument can choose another session, home, snapshot or arbitrary source
index.

```json
{"query":"literal phrase","limit":5,"offset":0}
```

`ariadne_find` searches case-sensitive literal substrings across archived
versions in the attached session. It returns event handles (`event:<sha256>`),
roles and clearly lossy ASCII-escaped previews, not complete records. Query size
is at most 1024 UTF-8/surrogatepass bytes; limit is 1–10. `count` is the page count,
not a total. `next_offset` is only a continuation candidate: mutation may shift
pages and an empty terminal page is possible. No `has_more` is invented.

```json
{"ref":"event:<64 lowercase hex characters>","offset":0,"limit":2048}
```

`ariadne_zoom` accepts validated event handles or generation-bound navigation
handles with this exact format:

```text
node:<checkpoint_sha256>:<projection_id>:<start>+<count>
```

Both digests are 64 lowercase hexadecimal characters. The checkpoint digest is
SHA-256 of the exact immutable checkpoint bytes selected by the validated
manifest, not a summary-text hash, summary version, projection ID or path guess.
`start` is a nonnegative decimal integer and `count` a positive decimal integer,
each at most ten digits without leading zeros. The unchanged kernel validates
projection identity and aligned, in-range spans after the wrapper validates the
attached generation. All returned node/child/leaf descriptors retain the same
checkpoint digest. Legacy `node:<projection_id>:<start>+<count>` handles are
refused, never rebound.

Nodes expand into bounded child descriptors; source leaves return exact
whole-record canonical-JSON byte pages. Sources remain session-scoped. A
republished checkpoint changes node handles even for the same source event IDs
and identical summary text when checkpoint settings differ. Loaded views stay
frozen until explicit lifecycle reload; unchanged checkpoint handles survive
restart and unrelated archive appends. Old-generation node handles fail after
loading a different checkpoint. Content-addressed `event:<sha256>` handles do
not depend on prepared generations and remain usable for unchanged sources.
Byte offsets apply to sources only. Page limits are 1–2048 bytes.

For framework-free hosts, `PreparedView.checkpoint_id` exposes the exact validated
manifest checkpoint digest. `handle_evidence_tool(..., kernel=view.kernel,
checkpoint_id=view.checkpoint_id)` takes it only as trusted host attachment state;
there is no caller-visible checkpoint/generation argument. The host must supply
the digest verified for that exact loaded kernel. Missing or malformed trusted
identity refuses node retrieval with `checkpoint_unavailable`, mismatches with
`checkpoint_mismatch`; neither error exposes source or summary data. Event
retrieval does not require a checkpoint. The prepared Hermes engine always
supplies its attached view's identity and ignores caller lifecycle/scope kwargs.

Every complete result, including envelopes and JSON escaping, is bounded to
8192 bytes when serialized as compact ASCII JSON. Canonical source pages use
base64, not lossy text clipping. Concatenate **base64-decoded bytes** in offset
order before UTF-8/surrogatepass decoding and JSON parsing; a page can split a
Unicode code point. Previews cannot reconstruct the original. The bound limits
outbound results, not archive scan cost or reader memory.

## Rollback and limits of evidence

```bash
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set context.engine compressor
```

Restart only the development process. Keep the canonical archive; no transcript
restore or source migration is required. Remove/move the prepared plugin directory
only when that process is no longer using it.

Pure tests validate artifact, mapping and bounded-codec contracts. Real-Hermes
loopback tests separately exercise discovery, transport and tool dispatch using
synthetic data and explicitly fixture-generated summaries. Neither establishes
long-horizon semantic recall, real provider caching or live-profile activation.
The earlier paid offline compactor proof is separate, not a live-recall benchmark.
