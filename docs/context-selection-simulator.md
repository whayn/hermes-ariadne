# Offline context-selection simulator

Status: development-only pure library, introduced for #4. It is **not invoked by
AriadneEngine**, is not copied by the runtime installer, does not read a Hermes
home, and makes no model or network calls. The live engine still inherits the
host's no-op `select_context`. Do not enable the passive engine on a live profile
expecting this module to compact it.

## Purpose and boundary

Prove the request-selection contract before implementing the UniiChat tree or
changing what Hermes sends to a provider. The simulator takes an explicit branch
projection and explicitly supplied summaries. It validates and materializes one
candidate; it does not decide which old material to summarize, create summaries,
merge tree nodes or choose a smaller frontier if the candidate is too large.

This is original contract/simulation code, not a port of UniiChat's algorithm.
The pinned algorithm reference remains in `integration-findings.md`.

## API

```python
from hermes_ariadne.simulation import SourceMessage, simulate_request

history = [
    SourceMessage("old-u", {"role": "user", "content": "Compare A and B."}),
    SourceMessage("old-a", {"role": "assistant", "content": "A is smaller."}),
    SourceMessage("prev-u", {"role": "user", "content": "Choose A."}),
    SourceMessage("prev-a", {"role": "assistant", "content": "Chose A."}),
    SourceMessage("now", {"role": "user", "content": "Why?"}),
]
result = simulate_request(
    [{"role": "system", "content": "Host-controlled instructions."}],
    history,
    retain_from=2,
    summaries=[{
        "node_id": "example-node",
        "event_ids": ["old-u", "old-a"],
        "text": "Compared A and B; A was smaller. Expand for evidence.",
    }],
    budget_units=2000,
    reserved_units=100,
)
assert result.status == "fits"
assert result.messages is not None
```

All values above are synthetic. Real source identities must come from the caller's
authoritative branch/revision projection; the simulator never invents them.
`SourceMessage` is frozen at the attribute level, but its nested message mapping is
caller-owned. Inputs must not be changed concurrently during a call. Returned
messages and the measurement callback receive independent deep copies, so edits
on either side cannot write through to the other.

### Inputs

- `prefix`: leading host-controlled system/developer mappings, preserved exactly.
- `history`: ordered `SourceMessage(event_id, message)` records for one explicitly
  selected branch/revision projection. Event identities must be unique and
  nonempty. Historical revisions remain archived elsewhere; this API does not
  resolve latest versions or duplicate observations automatically.
- `retain_from`: index of a user-turn boundary. It must retain the previous
  user-led exchange and the whole current user-led turn. Keeping more is allowed.
  Empty history uses zero. This explicit boundary is not inferred from assistant
  prose, timestamps or similar content.
- `summaries`: ordered mappings with `node_id`, `event_ids`, and `text`. Their
  concatenated event references must equal precisely `history[:retain_from]`, in
  order. Empty, overlapping, reordered, invented or missing references are rejected.
- `budget_units`: nonnegative integer maximum window. Zero means unknown, not
  infinite. `reserved_units` accounts for costs outside the measured message list.
- `measure(messages)`: trusted, deterministic whole-request counter. It must return
  a nonnegative integer. Its count is not assumed additive across messages. It is
  given an independent copy, but the caller remains responsible for its accuracy.

Sources must contain finite JSON values without implicit tuple/number-key
coercion. Unsupported roles or malformed retained tool pairing raise `ValueError`.
This is deliberately stricter than replaying arbitrary legacy/transformed host
rows: a future adapter must choose a valid projection, not repair it silently here.

### Measurement is not provider token accounting

The default `ascii_json_units` counts canonical **ASCII-JSON bytes**, including
metadata. It is deterministic test instrumentation, not an estimate of BPE tokens,
not the actual provider wire format, and not an assertion that this request fits
any model. Unicode escapes and metadata make its costs differ from provider input.

A future host adapter must supply/calibrate appropriate counting and reserve
system/tool-schema/output/provider overhead. Hermes's `budget_tokens` is the full
context length, not an already-discounted input allowance. A simulated fit under
an inaccurate callback does not prove a provider fit.

## Result and failure behavior

`SimulationResult` records:

- `status`: `fits`, `over_budget`, or `unknown_budget`.
- `messages`: a usable candidate only for `fits`; otherwise **None**.
- `used_units`: whole-candidate measurement, or None for an unknown budget.
- `available_units`: window minus reservations, floored at zero, or None if unknown.
- `archived_event_ids` and `retained_event_ids`: exact, ordered local audit coverage.
- `memory_text`: deterministic derived navigation text for diagnostics.

The candidate must satisfy `used_units + reserved_units <= budget_units`. A
reservation larger than the window is blocked, even if a counter returns zero.
Unknown budgets do not invoke the callback. Malformed inputs raise rather than
returning a partly repaired request. The simulator never truncates a recent
message, discards coverage, drops a tool result, or returns an oversized usable
candidate to make a test pass.

## Memory representation and tool fidelity

For this offline proposal only, a nonempty memory view is a synthetic assistant
message between the untouched host prefix and the retained user-led suffix. Each
summary is one canonical JSON line under an explicit untrusted-data header:

```json
{"node_id":"example-node","source_count":2,"source_start":0,"text":"..."}
```

Full event-ID lists remain local audit data, **not prompt text**. This keeps the
navigation representation small as a node covers more history. `source_start` is
a position in this supplied projection, not a global durable event identity;
`node_id` is the future lookup handle. No zoom tool exists for this simulator yet.

Summary text is JSON-quoted and is never interpreted as a new system/developer
role or a live tool call. This structural property is **not proof of prompt-
injection resistance**. Future runtime policy still must treat history as
untrusted, preserve permission boundaries and inspect original evidence.
Coverage validation does not establish that a supplied summary is semantically
accurate or that a model would remember the right thing.

Retained raw messages stay exact, including nested payloads, code, whitespace,
Unicode, metadata and tool arguments. The simulator validates pending tool-call
IDs, complete parallel results and duplicate/orphan/missing results. Reused IDs
are accepted across distinct completed occurrences, not within one parallel batch.
A provider-request boundary with pending results is rejected; intermediate
execution states before all results arrive are not model requests in this model.
Full provider-specific message schemas and ephemeral host-prefill variants are
not validated or certified here.

## Cache invariants actually tested

- Repeating the same inputs produces the same result.
- With a fixed view, extending the live tool loop preserves memory text and the
  already materialized raw prefix.
- Appending a supplied summary slice preserves prior memory lines byte-for-byte.

These are logical/text-prefix invariants. There is no tested batch-merging policy,
provider prompt-cache breakpoint placement, tokenizer-prefix guarantee or measured
cache hit rate. Recompression may legitimately change the prefix; UniiChat's
cache-stable merge algorithm is a later milestone.

## Verification

```bash
.venv/bin/python -m pytest tests/test_simulation.py -q
```

The fixtures are synthetic and defined in that test module: interleaved toy-project
and garden threads, exact code/Unicode, parallel tools, reused IDs, append/retry
sequences, invalid coverage and large protected messages. Tests sweep budgets and
check cold-prefix lengths while keeping the prompt budget fixed. No personal
transcript, real archive or fabricated provider answer is involved.

CI runs these pure tests on Python 3.11 and 3.14 and also runs the full pinned
real-Hermes suite, which still verifies that the live engine does not replace
context. This is not model-recall evaluation or production certification.

## Next milestone

Use this contract as an invariant checker for a faithful pure UniiChat tree/view
implementation with supplied deterministic summary fixtures. Then introduce
versioned summary persistence and a real compactor behind it. Only after the
selection algorithm and actual host/provider boundary pass should a later PR
wire `AriadneEngine.select_context` into live requests. Summarization quality,
proactive recall, imports and cross-session retrieval remain separate work.
