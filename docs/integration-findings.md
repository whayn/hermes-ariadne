# Ariadne v0.1: inspected contracts and decisions

## Evidence

Inspected installed `/opt/hermes` source (distribution stamp: 0.22.5,
bootstrap commit bf0203a794a07aac3ff764972ab9087baa4e4f4e), the available
managed workspace, upstream Hermes c627ec84664d43e0b18b95d9d86d8963747d56c7,
and UniiChat ff32319b089dc49c82c29f9a7af1f643eb10d43a.
The installed ContextEngine ABC matches upstream at inspection time. The stamp
is distribution metadata, not proof that every installed source file matches
that historical commit.

Official reference: https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin

### Hermes

- `agent/context_engine.py`: ContextEngine ABC requires name,
  update_from_response(usage), should_compress(prompt_tokens=None), compress(...).
  Directly read budget/token fields, update_model and reset defaults matter.
- `plugins/context_engine/__init__.py`: directory discovery accepts
  `$HERMES_HOME/plugins/<name>/__init__.py`, with register(ctx) calling
  ctx.register_context_engine(instance). Selection is `context.engine`; bundled
  names win. Import/discovery must have no storage side effects.
- `agent/agent_init.py`: selects the engine, updates its model, optionally calls
  bind_session_state(session_db, session_id), injects schemas and starts session.
  The binding seam is duck-typed host integration, NOT an ABC method; test it.
  General-plugin shared instances require clone_for_agent().
- `agent/conversation_loop.py` and `agent/turn_finalizer.py`: on_turn_complete
  observes structural clones after persistence in normal finalization. Exceptions
  are logged and swallowed; abnormal early returns can bypass this callback.
- `run_agent.py`: actual session end/reset/start, not per-turn end. Per-agent
  state must not leak across resets, model changes or subagents.
- `agent/message_metadata.py`: message_uid is logical identity, not _row_id.
  Copies/branches preserve it; content can change under it. Merges keep a uid
  plus _absorbed_message_uids. Tool ids can repeat; _tool_call_uid(s) preserve
  occurrence identity. Missing uid means not identified yet, not an error.
- `hermes_state_messages.py`: get_messages(session_id, include_inactive=True)
  returns decoded durable row snapshots without alternation repair. Prefer
  this over replay projection (which can strip/transform content) when bound.
  Read only the explicitly attached session, never scan a profile's old chats.
  Durable content is authoritative over injected/model-visible text.
- Tool schema injection is gated by context_engine toolset; handle_tool_call
  returns a JSON string. No global tool registry edits needed.
- Default select_context returns None before cache-control and sanitization.
  Leave it inherited. v0.1 does NOT implement compaction or replacement.

Testing conventions inspected: upstream tests/conftest.py isolates homes before
collection, removes inherited credentials/session environment and pins timezone;
tests/plugins/test_context_engine_user_dir.py exercises real discovery/selection;
tests/agent/test_context_engine_message_uid.py covers restore, rewrites, merges
and tool occurrences; host-contract/turn-complete tests cover lifecycle/binding.
The standalone suite must isolate environment BEFORE importing Hermes too.

### UniiChat

- Memory.ts main JSONL messages are authoritative; tree summaries and fixed-slot
  indexes are separate. Reindex recovers logs and cuts incomplete trailing lines.
  Acts have durable idempotency keys. Images live separately by content hash.
- Binary nodes (level,index) cover original ranges; 512-byte summaries recursively
  merge children. zoom expands siblings or returns original paginated content.
- Two views cover chat and summary-builder contexts. Ready sibling merges happen
  in batches (128KB to half for chat), preserving a stable prefix between batches.
  Views emit four-line blocks with one explicit cache breakpoint.
- AskAI.ts provides provider dispatch, patient retries, aborts and cache-prefix
  single-flight coordination. Do NOT port its agent/provider loop into Hermes.
- spec/memory.test.ts rebuilds indexes of an old chat and compares exact view;
  cache.test.ts runs real memory/agent code against a simulated caching provider;
  failsafe.test.ts distinguishes vendor outage from input refusal.
  These were inspected, not executed as Ariadne tests.

## Implementation increments

1. Append-only SQLite versioned snapshots + provenance, stable host uids,
   deterministic event ids, atomic idempotent ingestion; unit tests first.
2. Passive ContextEngine + exact literal search, session-scoped by default;
   reconcile the attached durable session on start/turn/end. Separate connections
   per transaction, fresh clones. No provider changes, no selection/compaction.
3. Isolated real Hermes discovery/agent/tool dispatch and restart tests using a
   deterministic local provider fixture; packaging, install/rollback docs.

## Risks and explicit boundaries

This is an observation foundation, NOT a complete lossless-memory claim. A host
mutation before any observation cannot be recovered; missed callbacks are healed
only when that same durable session is next reconciled. Missing legacy uids are
skipped. Never invent uids from content or positions. Observe all retained rows,
including inactive/rewound rows: rollback in Hermes does not erase this archive.
Snapshots can contain secrets and provider reasoning. No encryption/redaction,
attachment-byte copying, cross-profile access or background scanning in v0.1.
Store versions, not latest-only overwrites; search returns historical versions.
Full rescans and literal search favor correctness over large-history performance.
Without compaction, near-window and overflow behavior is NOT production-ready.
No live profile activation; only disposable homes and environments are changed.
