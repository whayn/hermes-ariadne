# Prepared-prefix live ContextEngine bridge Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. This is an implementation plan for the user's execution request, not a plan-only handoff.

**Goal:** Ship an explicitly opt-in `ariadne-prepared` engine that uses an already-ready immutable archived prefix in real Hermes requests and provides bounded session-local evidence tools.

**Architecture:** Keep the existing `ariadne` engine/passive installer behavior unchanged. A separate engine loads one prepared snapshot at session start, validates its exact revisions against current active durable history and the assembled request prefix, and returns an isolated replacement only on supported mappings and successful policy checks. Frozen projection references survive appended raw tails; changed/rewound/mismatched prefixes decline selection rather than rebuilding or pretending to block host dispatch.

**Tech Stack:** Python stdlib archive/kernel/JSON; installed Hermes ContextEngine ABC; existing real-Hermes loopback integration fixtures; pytest/Ruff/build/actionlint/Gitleaks and hosted CI.

## Audited contract and non-goals

- Host hook: `select_context(request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0) -> list | None`.
- No ContextSelectionRequest/Result DTOs, supplied schemas/tokenizer/output reservation, or supported dispatch-veto result exist. `None`, invalid results and exceptions fall back to original messages. Never mutate inputs, raise to claim refusal, or claim a declined selection prevented sending.
- Budget checks cover bounded historical bytes and an explicit message-request byte policy plus a native rough-message estimate/operator reservation. They are NOT exact provider-token/full-tools/output/currency guarantees. Status must distinguish these units and mark full provider fit unverified.
- Preserve assembled host system/developer messages and the retained wire suffix. Raw durable bytes may differ from provider wire bytes. Reject unsupported replay/prefill/content transformations rather than guess.
- The existing directory loader's collector has no ctx.llm. No automatic live summarization, native SDK hooks, unbounded paid work, dynamic rebase, private backfill, cross-session/profile data, normal-profile activation or core changes in this milestone.

## Task 1 — prepared snapshot and pure selector (worker-owned)

**Create:** `src/hermes_ariadne/prepared.py`, `tests/test_prepared.py`.

Public interface:
- `publish_prepared(kernel, *, home, history_bytes=8192, request_bytes=65536, reserve_tokens=4096) -> Path`: operator API only; no model calls. Require kernel archive to equal `home/ariadne/events.sqlite3`, exact session scope, valid positive limits. Publish a private atomic manifest/checkpoint beneath `home/ariadne/prepared/<sha256(session_id)>/`, reject unsafe paths/symlink files and protect canonical archive. Existing immutable generations must not be silently repointed by tool arguments.
- `PreparedView.load(store, session_id, home) -> PreparedView | None`: absent manifest means disabled; validate format/settings/session/projection/checkpoint digest and load an existing checkpoint without manufacturing ready state. Expose `.kernel`, `.projection` for tools. Freeze this snapshot for session lifetime; no mid-tool-loop refresh.
- `.select(request_messages, *, conversation_messages, incoming_message, active_messages, budget_tokens, measure_tokens) -> SelectionDecision`, with `.status`, `.messages`, `.diagnostics`.

Selection rules:
1. Exact archived events form an ordered prefix of authoritative active durable rows, including logical UID/revision equality; never infer chronology from archive hash/order. Require a bound active source, no duplicate/absorbed/ambiguous source identities.
2. Prepared prefix also corresponds conservatively to the current conversation prefix and assembled provider prefix. The assembled request has leading host system/developer messages then a supported one-to-one history shape. Verify removed prefix provider fields exactly; unsupported historical api_content/reasoning/prefill transformations decline. Do not substitute raw DB content for retained wire messages.
3. The frozen prefix ends on a user-turn boundary and leaves the previous exchange plus current incoming turn and all subsequent tool messages raw. Preserve complete parallel tool groups and reused call-ID occurrences. Detect invalid source/input shapes without mutating them.
4. Only fully ready contiguous kernel coverage becomes an assistant-role untrusted navigation block. References derive from the frozen projection, not the growing tail. Do not fabricate missing summaries, clip originals, mutate kernel private state or change authoritative history.
5. Measure the entire replacement message list and bounded historical block. Unknown window, failed/native rough estimates, insufficient explicit reserve, excessive request bytes/history bytes/protected suffix return messages=None and observable decline reasons. Report rough/message-only scope and provider_fit_verified=false.

TDD vertical slices: absent/valid publication -> exact loaded prefix -> appended tail selection -> revision/rewind/foreign/incomplete rejection -> protected tool/current turn -> measured-policy decline -> stable references/deep-copy isolation. Tests must witness RED before each behavior. No Hermes/provider imports in this module.

## Task 2 — bounded evidence tools (independent worker-owned)

**Create:** `src/hermes_ariadne/retrieval.py`, `tests/test_retrieval.py`.

Public interface: `tool_schemas() -> list[dict]`; `handle_evidence_tool(name, args, *, store, session_id, kernel=None) -> dict`.

- Names `ariadne_find` and `ariadne_zoom`; no change to legacy `ariadne_search` in passive engine.
- `ariadne_find`: literal query, bounded descriptor/clearly-labelled preview results, immutable event handles (`event:<event_id>`). Never return entire messages. Validate types (reject bool), query/limit/offset. Explain offset pagination is stable only while archive observations are unchanged; no invented total/has_more.
- `ariadne_zoom`: exact event canonical JSON-byte pages or binary kernel node expansion (`node:<kernel.reference>`). Event lookup always uses bound session. Node handles require supplied kernel with matching session/projection; foreign/stale handles fail. Fixed response-envelope budget 8192 ASCII-JSON bytes and page limit <=2048 bytes; previews labelled lossy while base64 canonical pages are exact, including split Unicode bytes. Node expansion must not leak arbitrarily large summary text.
- Every result is labelled untrusted historical evidence, never instructions. Caller cannot choose session, filesystem path, snapshot or source index independently of a validated handle. Return bounded error results too. Original records untouched.

TDD: large Unicode/multiline exact page reassembly, descriptor bounds, final/no-next page, empty search, literal escapes, historical versions, cross-session refs, malformed args, huge summaries, missing kernel, immutable input and total serialized bound.

## Task 3 — opt-in engine and packaging (parent-owned)

**Create:** `src/hermes_ariadne/live_engine.py`, `prepared_entry.py`, `prepared_plugin.yaml`, `tests/test_live_engine.py`.
**Modify:** `install.py`, `pyproject.toml` package data, `tests/test_install.py`.

- `PreparedAriadneEngine(AriadneEngine)` name `ariadne-prepared`; default passive class/registration/schema remains unchanged.
- New static plugin entrypoint registers the prepared class; no config/DB/model I/O on import or construction. Installer `--prepared`/`install(..., prepared=True)` explicitly selects separate plugin directory/name and copies only required stdlib runtime files. No native compactor, receipts SDK code or caller environment is installed. Preserve default four-file install and atomic refusal/cleanup behavior.
- Session start observes exact bound archive, loads optional prepared snapshot, initializes diagnostics. Binding/reset/clone/model changes must not leak session/projection state. Load once; explicit absence/unavailable state declines selection without claiming hard refusal. Snapshot session must match current bound session on every use.
- Select reconciles archive, reads attached active durable history, calls pure selector with native `estimate_request_tokens_rough` (late import), and records honest diagnostics. No model requests or SDK hooks. Return candidate or None without mutating request/conversation/tool inputs.
- Expose only the two bounded evidence schemas in prepared mode, available before session initialization; real host context_engine toolset gating remains authoritative. Dispatch returns compact JSON strings and treats supplied live messages read-only.

TDD: default class unchanged -> prepared session lifecycle -> installed discovery -> hook candidate/decline and exact fallback -> clone/reset/foreign binding -> tools work before/after snapshot availability.

## Task 4 — real Hermes vertical tracer and restart (parent-owned)

**Create:** `tests/fixtures/prepared_worker.py`, `tests/test_live_transport.py`.

Extend the established subprocess/loopback testing approach, not a replacement agent harness. Seed synthetic durable history, archive exact rows, build clearly fixture-generated ready summaries and explicitly publish a prepared snapshot before real agent initialization. Run installed directory discovery, AIAgent conversation/request assembly/sanitizers, actual SDK HTTP transport and engine-tool dispatch against a scripted loopback server.

Capture real outgoing payloads and compare against a passive/no-op baseline: old raw prefix absent, stable untrusted navigation handles present, system/protected previous/current/tool wire suffix unchanged. Script find -> node zoom -> exact source pages, reassemble original code and compare canonical hashes. Verify raw durable old rows unchanged and restart recovery. Exercise disabled tools, bad snapshot/revision, unknown/oversized policy and actual None/invalid/exception fallback semantics; no misleading dispatch-veto assertion. No real credentials/network outside loopback or private history.

## Task 5 — review, docs, CI and artifact

- SPEC then independent quality review for each worker module; tight affected tests during fixes. Parent owns final canonical full gate; no repeated thousand-round store stress from workers.
- Add pure tests to both archive CI lanes and all tests to unfiltered real-Hermes lane; explicit existing SDK compatibility baseline/override disclosure unchanged.
- Document operator preparation, explicit new engine install/config/rollback, supported strict wire mappings, frozen-prefix growth limitation, bounded evidence units, fail-open host and lack of automatic/proactive summarization. Update AGENTS only for actually shipped prepared slice; passive remains passive.
- Lint/format, package wheel+sdist contents and installed prepared discovery, actionlint if workflow changed, secret scan, real-host trace/restart, full hosted suite and actual current-head Codex review before calling ready. No private corpus or default activation.

## Risks / escalation

If real request assembly cannot establish the strict source-prefix association, decline that shape rather than guess and narrow documented support. If a hard dispatch guarantee or automatic ctx.llm preparation becomes necessary, stop and design an upstream-supported seam separately; do not smuggle it into the offline guarded adapter or mutate shared client state. This slice is not indefinite bounded live memory or semantic-recall certification.
