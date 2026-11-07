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
Nonempty reasoning-bearing prefix sources are refused at publication, loading
and selection; archived originals can still retain them exactly.

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

A prepared view (or its absence) is loaded once per attachment. The initial bind
before session start performs no archive I/O. After initialization, reset/end
followed by binding, or binding another session/DB, starts a new attachment using
the remembered explicit profile home; it does not require another start hook.
Repeated binding of the same attached session/DB does not reload even an absent
view. Explicit session start can reload or select a different profile; clones
start with no attachment or remembered home. Republishing during a tool loop does
not change the view. Model switches invalidate the last measurement, not the
frozen source references; the next request uses its newly supplied window.

Operations capture their attachment generation and source scope. If a lifecycle
change occurs during a read, stale scans are discarded before archival ingestion,
stale selection declines, and stale tools return `lifecycle_changed` without
historical evidence. This fences work that has already entered its hook; Hermes
supplies no caller lifetime token with which to identify an old request that
first enters a hook **after** rebinding. It is not a universal concurrent-request
or transport-cancellation guarantee.

“Frozen” describes the live engine's attachment policy, not deep immutability of
arbitrary Python objects. Direct Python users must not build, save or mutate a
loaded `PreparedView.kernel`; use a separate working kernel to prepare another
publication. The direct evidence helper's `kernel` and `checkpoint_id` are a
trusted, matching attachment pair, not model-controlled arguments. Passing a
mismatched pair or mutating that kernel violates this API precondition. The live
engine captures both from the same private prepared attachment and never builds
it; tool arguments and caller-supplied attachment overrides cannot replace them.

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
multimodal messages, replay/thinking carriers outside the finite contracts below, changed `api_content`,
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

Independently assigned native Codex tool-item IDs are supported when they are
clean unsigned ASCII `fc_`-prefixed values within 64 characters and the call's
`call_id` agrees with its `id`. Unlike deterministic sidecars, a non-derived
native item ID remains in comparison identity: all available canonical,
restored and assembled views must retain the same value. Missing or swapped
IDs, changed argument bytes, signatures and unknown call fields decline.
Within each single assistant `tool_calls` array, duplicate IDs admitted through
this **non-derived native-item path** also decline before candidate measurement,
even when all three views agree. The comparison resets this check for every
message: reuse across assistant messages remains supported with distinct
occurrence UIDs, for the same or different provider call IDs. This is a
response-local supported-envelope admission rule, not history/session-wide
uniqueness, issuer provenance or provider certification. Values matching the
host's deterministic derivation still follow the legacy sidecar path, including
repeated derived IDs and mixed derived/independent collisions. The available
fields cannot distinguish a native ID equal to that derivation or establish a
response/issuer namespace across historical messages; this rule does not resolve
that ambiguity by imposing a broader uniqueness policy.

Parallel/reused call-ID occurrence checks still apply. This concerns the
assembled selection-hook message: Hermes's native formatter subsequently omits
that item ID from HTTP function-call items while preserving the call/result
link. Ariadne neither changes that formatting nor rebuilds the retained suffix.

One ordinary native Codex final-answer carrier is also supported: exactly one
completed assistant `message`, containing exactly one nonempty `output_text`
part matching the canonical answer. Its optional ID must be a clean ASCII
`msg_`-prefixed value within 64 characters; its phase must be absent or
`final_answer`. The complete validated item (including ID, status, phase and
untrimmed text) remains part of comparison identity. Hermes stores this carrier
as JSON text in canonical durable rows and restores it as a list. Only canonical
active rows receive strict comparison-only JSON decoding; duplicate keys,
nonfinite values and malformed shapes decline. Request/replay strings are not
silently decoded. Source hashes still cover the exact stored representation,
and the actual retained assembled suffix is not rewritten. Commentary,
multiple-message outputs, signatures and unknown or contradictory fields remain
unsupported. The additional retained-tail reasoning contract is below.

### Source-bound encrypted reasoning in the recent tail

Some ordinary answers and archive lookups also carry provider-issued sealed
continuation state in `codex_reasoning_items`. This is not a user memory store.
Ariadne does not decrypt, summarize, regenerate or strip it. The experimental
[issue #14](https://github.com/whayn/hermes-ariadne/issues/14) implementation admits
one finite envelope on an otherwise supported **retained assistant answer or
tool-call message**, while returning the actual assembled recent messages
unchanged. Native continuation/restart and installed-artifact verification are
separate gates; the pure checks alone do not establish those results.

The item must have exactly six keys: `type`, `encrypted_content`, `id`, `summary`,
`_issuer_kind` and `_issuer_model`. Its type is `reasoning`; ciphertext is nonempty
ASCII, preserved without decoding or interpreting it. The ID must be a clean
ASCII `rs_` value within 64 characters, excluding `rs_tmp_` IDs. Summary parts are
ordered exact `{type: "summary_text", text: str}` objects, including empty lists
or text. All new text must be valid UTF-8. Unknown fields, unrelated nonempty
`reasoning_details`, unsupported shapes and missing issuer/model stamps decline.
Both stored JSON text and decoded list envelopes have a separate 1 MiB admission
bound; strict canonical decoding rejects duplicate keys and nonfinite values.

Every admitted reasoning-bearing row must have canonical active-source coverage
and exact UID/session/order correspondence. Its complete encrypted item must agree
across canonical JSON text, restored conversation lists and assembled request
lists. Repeated reasoning IDs across different retained rows in one candidate
request decline because the native formatter can suppress repeated item IDs.
This is a request-local reasoning rule, not a global/session ID namespace or a
change to existing function-call occurrence semantics.

Readable companions are checked independently rather than inferred from the
summary or merged. Both canonical SQL columns `reasoning` and `reasoning_content`
must exist and be NULL or UTF-8 strings. Each field follows this exact mapping:

| Canonical value | Supported restored value |
| --- | --- |
| NULL | key absent or explicitly `None` |
| Empty `reasoning` string | exact empty string, or absent only with `_db_persisted is True` |
| Empty `reasoning_content` string | present exact empty string |
| Nonempty string | present exact string, including whitespace |

The two fields may contain different strings. The assembled message must omit
**both readable keys entirely**, matching native assembly; even a present NULL or
empty value declines. This comparison-only omission never rewrites the archive
or conversation. Missing canonical coverage or a one-view mutation declines.

The live engine needs complete metadata from Hermes's real `update_model`
callback: exact provider `openai-codex`, API mode `codex_responses`, a nonempty
base URL/model, native issuer classification `codex_backend`, and the exact
native-normalized bound model. Native xAI/profile precedence applies in the
callback's Hermes context. Missing helpers, incomplete/unsupported bindings and
safe-worker profile bypass disable reasoning permission. No API key is inspected
or retained, and no raw base URL is retained. Direct Python callers of
`PreparedView.select` have no permission by default; its optional
`ReasoningReplayPolicy` is a trusted host argument, not a model-controlled tool
parameter.

Each model update advances a separate route revision, even for identical values.
Selection fences callback-capable reads, measurement and result/diagnostic
publication against that revision as well as its session attachment generation.
A stale classification cannot overwrite a newer binding. Model updates do not
refresh the frozen prefix. Clones start without reasoning permission; same-agent
reset/rebind may retain the latest explicit route binding.

**Issuer/model stamps are labels, not authentication.** This contract establishes
source correspondence plus the latest bound-route label match—not endpoint or
account authenticity, decryptability, provider acceptance, final physical-wire
model identity or context fit. Later request/model/input overrides and recovery
remain Hermes-owned. A selection decline leaves the original request available
to Hermes; it does not veto dispatch or change native filtering.

Reasoning-bearing **older prefixes remain unsupported** at publication, load and
direct selection. Keep those messages in the raw recent tail; this milestone does
not compact reasoning state or automatically advance the prepared frontier.

Canonical stored carrier JSON has a separate **1,048,576-byte UTF-8 admission
limit**, including any JSON whitespace. Selection first checks character length,
then encoded length before strict JSON decoding; oversized carriers decline
without rewriting history. This is not the candidate `request_bytes` allowance,
a whole-source/process-memory bound, or a dispatch veto: Hermes can still send
the original request after a decline. Original strings and source hashes remain
unchanged, and no decoded-carrier cache survives a selection.

Legacy empty-as-absent markers (`None`, `""`, `False`, `[]`, `{}`, `0`, `0.0`)
keep their previous comparison semantics. Hermes normally persists these as SQL
NULL and omits them from native replay. They are not nonempty replay payloads
whose identity was dropped. Nonempty canonical JSON strings such as `"[]"`,
`"false"`, `"{}"`, `"null"` or `"0"` are not this exception: they fail the strict
supported-carrier checks. Nonempty carrier omission or substitution between
canonical, restored and assembled views still declines.

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
not a total. Each index candidate is re-read through the hash-checked,
session-scoped event lookup and its literal match is rechecked against the
canonical content's exact indexed text/JSON representation. An unavailable,
corrupt or stale-index candidate refuses the whole page with
`evidence_unavailable`; no partial descriptors or misleading pagination are
returned. `next_offset` is only a continuation candidate: mutation may shift
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

## Native Codex transport regression

`tests/test_codex_transport.py` uses actual `AIAgent`, `SessionDB`, installed-plugin
discovery, OpenAI SDK HTTP and Responses SSE with an isolated custom loopback
endpoint in `codex_responses` mode. Replies and summaries are explicitly scripted
synthetic fixtures, not real model output or a connection to OpenAI's Codex
backend. Ordinary replies, actual find/zoom tool iterations, subsequent user
turns and a separate-process restart must all select the prepared view.

The tests independently check the candidate and actual HTTP payload: navigation
must arrive first and exactly once, using Hermes's audited send-time outer-text
whitespace trimming rather than raw hook bytes; original protected text and host
instructions must match the source-validated no-op baseline; native carrier values
and durable call/result occurrence UIDs must survive replay and restart. Exact source pages
are reassembled and hash-checked separately from model-like output. Pure mutation
probes ensure the assertions reject missing/corrupted navigation, changed
protected text/instructions, null/regenerated occurrence UIDs and a bad baseline.
The actual provider pilot remains a separate opt-in verification step.

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
