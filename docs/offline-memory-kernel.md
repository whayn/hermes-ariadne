# Offline archive-backed memory kernel

Related scope: [issue #6](https://github.com/whayn/hermes-ariadne/issues/6).

This is an **offline development module**, not a live context engine. Installed
Hermes behavior remains the passive v0.1 archive. Do not enable that engine on an
important profile expecting these modules to compact its context.

## Reproduce the vertical slice

Use the repository's separate development environment and a NEW directory:

```bash
.venv/bin/python -m hermes_ariadne.demo "$TMPDIR/ariadne-kernel-demo" --phase build
.venv/bin/python -m hermes_ariadne.demo "$TMPDIR/ariadne-kernel-demo" --phase recall
```

The first process archives a synthetic, interleaved conversation containing an
old, multiline code fragment. It builds the tree with deterministic topic labels,
compiles a summary view and passes a historical prefix into the existing offline
selection simulator, retaining the recent suffix unchanged. The second process
loads persisted state, follows the code marker from the view through child nodes,
and reassembles paged canonical source bytes from SQLite. It does **not** ingest
messages, regenerate the code or call the summary provider. The persisted
expectation contains only source identity and a hash, not a source copy.

Reports are JSON on stdout. Successful recall reports an exact code hash match,
a navigation trace, multiple pages, distinct process id and unchanged archive
row counts. This demonstrates mechanics and persistence, **not model recall or
summary quality**. All content is synthetic; there are no provider credentials,
network calls or Hermes imports.

## Source truth and projections

`Projection(archive, session_id, event_ids)` requires an explicitly ordered list
of exact archived versions within one selected session. It rejects duplicate
event identities and multiple revisions of the same logical `message_uid`.
Neither hashes nor observation positions are treated as an edit chronology; it
does not infer a latest version or combine branches. The source sequence is
immutable: choose a new projection/state file for another revision or branch.

`EventStore.get_event` reads exact source mappings through session-scoped
membership and checks their SHA-256 identity. The archive's schema/version is
unchanged. Only snapshots already observed by Ariadne are recoverable; external
attachment bytes and pre-observation mutations are still outside its guarantee.

`MemoryKernel` binds that projection to a separate derived JSON snapshot. Source
identities, settings, ready summary texts and the chat/compactor frontiers are
persisted. Parent ranges deterministically map to adjacent child ranges; leaves
map to the original event identities. Sources are never replaced by summaries.
The state file is disposable/rebuildable, format/version checked, rejects foreign
files/settings/projections and is written through a private temporary file plus
atomic replacement. It rejects the archive file and SQLite's reserved companion
paths (`-wal`, `-shm`, `-journal`), including resolved directory aliases. Loaded
views may contain pending leaves, but never a missing internal summary node.

This first offline implementation assumes **one writer**. It checkpoints at build
boundaries, not after every model job; process termination can lose derived work
since the previous checkpoint. It is not a durable concurrent compaction queue.
It loads the derived catalog into RAM and is not yet performance-certified on
large histories. Do not copy derived-state storage into a future paid worker
without first addressing job receipts, incremental persistence and concurrency.

## What is ported from UniiChat

Reference: Victor Taelin / Higher Order Company's
[`Memory.ts` at 0500bf6a8ea4bf0ce047488562022b1b6532cc30](https://github.com/VictorTaelin/uniichat/blob/0500bf6a8ea4bf0ce047488562022b1b6532cc30/Memory.ts).
See [NOTICE](../NOTICE) and [LICENSE](../LICENSE).

The pure Python module implements range math, forest roots, frontier reach,
adjacent ready-parent merging, reference priority/older tie-breaking and delayed
batch decisions. The oracle fixtures execute selected original upstream
functions, not a separately handwritten implementation. The fixture generator
records its extraction procedure and reference revision. Tests run offline from
checked-in fixtures; regenerating them requires the pinned source and Node.

It is **not a full UniiChat port**. There is no UI, Bun runtime, provider client,
worker pool, filesystem compatibility or provider-specific prompt-cache mapping.
Ariadne retains the derived catalog instead of copying upstream RAM eviction.
The kernel runs dependency-ready jobs sequentially, ordered by source-context
cutoff and coarser level. This is a deterministic offline scheduler, not upstream's
asynchronous worker semantics.

## Bounds, pending work and source retrieval

- A node summary is nonempty UTF-8 and at most `node_bytes` (default 512 bytes),
  never silently clipped. A callback may return `None` to leave it pending.
- Leaves can use the full escaped source JSON if already small enough; otherwise
  the callback gets that source as input. Parent input is its two ready children.
- A leaf's context ends before that source; a parent's context ends no later than
  its exclusive range end. Both are capped by the earliest unbuilt leaf. Context
  is assembled only from ready parts ending at/before that cutoff.
- Missing parents are not invented. A view containing pending parts or exceeding
  its raw-summary byte bound is explicitly **unusable**, not presented as a fit.
- Reference view accounting measures **raw summary UTF-8 bytes**, excluding range
  labels/wrappers. This is NOT a complete request size or provider-token budget.
  The simulator separately measures the complete synthetic provider proposal.
- `SummaryRequest` is an offline callback boundary, not an API request. A large
  leaf input is still large. A real provider adapter must add input chunking,
  full request/token accounting and failure/cost controls before making calls.
- Node references include the projection identity. A foreign projection's handle
  cannot be used accidentally on another projection.
- `zoom` on a parent returns two child descriptors, including pending readiness.
  On a leaf it reads a page of **exact canonical JSON bytes from the archive**.
  These byte pages are base64 encoded: concatenate decoded chunks before decoding
  `utf-8` with `surrogatepass`, then parse JSON. Page boundaries may split Unicode
  codepoints. No large record or code fragment is silently cut off; `next_offset`
  explicitly indicates more bytes. The content field, whitespace, unusual Unicode
  and structured metadata are recoverable through the original mapping.

The base64 page codec is an internal exactness interface, not yet the final
human/model-facing text reader. A future tool should offer convenient text views
without losing the canonical-byte escape hatch.

## Next stages

1. Real compactor through Hermes's configured model routing, measured complete
   requests, durable job receipts and bounded retries; not a new model stack.
2. Isolated live `select_context`/`zoom` integration, preserving host ownership of
   tool execution, recent turns and provider dispatch.
3. Proactive and cross-session recall with explicit scoping and evidence handles.

No default profile config, host `state.db`, live tools or Hermes core is changed
by running the kernel demo.
