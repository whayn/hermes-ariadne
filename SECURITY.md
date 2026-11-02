# Security policy

Ariadne is experimental. There is no stable release or guaranteed response time.
Security fixes target the current development branch; older snapshots are not
maintained separately.

## Report privately

Use [GitHub private vulnerability reporting](https://github.com/whayn/hermes-ariadne/security/advisories/new).
Do not put exploit details, credentials, private conversations or archive files
in public issues or pull requests. If the private form is unavailable, contact
[@whayn](https://github.com/whayn) to arrange a private channel before sharing details.

Include the affected revision, impact, minimal synthetic reproduction and any
suggested mitigation. Never send a real memory archive to demonstrate a bug.

## Current boundaries

- Archives can contain private messages, secrets and tool payloads. They are not
  encrypted or automatically redacted.
- Hermes rewind/delete does not erase Ariadne's retained copies. There is no
  erasure command yet.
- Archived content is untrusted data, not a source of tool-use authorization.
- v0.1 is a passive archive, not a safe replacement for live context compression.
- Tests must use disposable homes, synthetic data and loopback provider fixtures.

See the README for the complete current limitations. Do not infer a privacy or
losslessness guarantee from an installation succeeding or tests passing.
