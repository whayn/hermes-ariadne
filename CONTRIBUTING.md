# Contributing to Ariadne

Ariadne is a small, experimental Hermes **ContextEngine plugin**. Correctness,
archive integrity and reviewable changes come before feature count. Read the
[README](README.md) for shipped behavior. Ordinary `ariadne` remains passive;
the separate `ariadne-prepared` engine consumes an operator-published frozen view.
Neither generates automatic live summaries.

## Before starting

- Search open and closed issues/PRs and inspect the current implementation.
- Open a feature/design issue before substantial behavior, schema, context-selection
  or public-tool changes. Agree on scope before a large implementation.
- Small documentation fixes and obvious, narrowly scoped bugs do not need a prior issue.
- One underlying problem per PR. Do not bundle unrelated cleanup or new frameworks.
- Never include private conversations, real memory archives, credentials, or live
  configuration. Reproduce problems with synthetic fixtures in disposable homes.
- Report vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Branches

Work from an up-to-date `main`; no permanent `develop` branch:

```text
type/short-kebab-description
type/123-short-kebab-description
```

Examples: `feat/cache-stable-view`, `fix/42-archive-write-failure`,
`docs/development-setup`, `ci/hermes-compatibility`.

Use the same types listed below. Issue numbers are optional. Contributor branches
follow this convention; Dependabot's generated branches are exempt. Delete merged
branches. Never force-push `main`; coordinate before rewriting a shared PR branch.

## Commits and PR titles

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
type(scope): short imperative description
```

The scope is optional; use an existing subsystem when it helps (`archive`,
`engine`, `install`, `ci`, `docs`, `deps`, `repo`). Allowed types:

| Type | Meaning |
| --- | --- |
| `feat` | New behavior |
| `fix` | Correctness or reliability fix |
| `docs` | Documentation only |
| `test` | Tests and fixtures |
| `refactor` | Restructuring without intentional behavior changes |
| `perf` | Measured performance improvement |
| `build` | Packaging or build tooling |
| `ci` | CI workflows and checks |
| `chore` | Other maintenance |
| `revert` | Revert a previous change |

Examples:

```text
feat(engine): add bounded context selection
fix(archive): retain message revisions across retries
test: cover interrupted archive transactions
chore(repo): establish public contribution workflow
```

Keep the subject at most 72 characters, without a trailing period. Use `!` before
`:` and a `BREAKING CHANGE:` footer for incompatible public API, configuration or
archive changes, with migration/recovery notes. Explain why in the body when not
obvious. `Refs #123` links work; `Fixes #123` means it is actually resolved.

Do not add `[verified]`, model names, emoji or invented provenance to titles.
Evidence belongs in the PR body. Local commits should follow the format; CI checks
PR titles, not every intermediate commit, because PRs are squash-merged.

Configure Git identity locally, using your own name/account and verified GitHub
noreply email. Do not adopt another contributor's identity or change global Git
settings as part of a contribution. Maintainer-authored work in this checkout uses
`Whayn <64968114+whayn@users.noreply.github.com>`.

## Development and verification

Use a separate development environment. Do not install into or modify a managed
Hermes runtime. See the README for installation and host-test setup.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -e .
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m pytest tests/test_store.py tests/test_crash.py tests/test_simulation.py tests/test_tree.py tests/test_kernel.py tests/test_kernel_demo.py tests/test_receipts.py tests/test_compaction.py tests/test_compactor_profile.py tests/test_host_site.py tests/test_prepared.py tests/test_retrieval.py -q
.venv/bin/python -m build
git diff --check
```

For engine, host lifecycle, discovery, installation or tool changes, run the full
suite with an inspected Hermes source and matching Python minor version:

```bash
HERMES_SOURCE=/path/to/hermes HERMES_PYTHON=/path/to/hermes/.venv/bin/python \
  bash scripts/test.sh -q
```

Tests must redirect `HERMES_HOME` before importing Hermes. Use the caller's scratch
`TMPDIR`, not a live profile. Loopback provider responses are scripted fixtures;
they are not proof of actual model recall. Add a regression test for a bug and,
where practical, demonstrate that it fails on the base revision and passes on the fix.

Hosted CI runs lint/format/build, archive/simulator/kernel tests on Python 3.11 and 3.14, a full
integration suite against a **pinned real Hermes revision**, and a history secret
scan. The host pin is in `.github/workflows/ci.yml`; one pinned revision is not a
claim of compatibility with all Hermes releases or providers. Hosted runners are
Linux; other operating systems are not certified by this workflow.

If workflows change, run `actionlint`. Record exact commands and actual results,
and explain anything not run. Do not weaken or skip a failing test to get green CI.

## Pull requests and review

Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md): describe the problem,
changes, verification, risks/rollback, and related issue. Mark incomplete work as
a draft, and disclose missing verification. A draft is not a promise of acceptance.

- Include tests and docs with behavior changes.
- Explain compatibility, schema/migration, privacy and performance implications.
- For a context compiler, explicitly verify budget bounds, source coverage,
  tool-call/result integrity, recent-turn fidelity and stable prompt prefixes.
- AI-assisted contributions are welcome. The author owns the result. For material
  agent-assisted work, state the model/harness and which checks were actually run;
  do not paste private prompts or fabricate review/testing claims.
- Address review feedback; request re-review after substantive changes.
- Maintainers squash-merge to `main` using the conventional PR title. CI must pass
  and review threads must be resolved. No minimum approval count is imposed while
  this is a solo-maintainer project; that avoids requiring an impossible self-review.
- After initial repository bootstrap, normal maintainer changes also use PRs.

For substantive code PRs, request a Codex review with a PR comment containing
`@codex review`, when the repository's Codex cloud integration is available.
Repository-specific rules live in the root `AGENTS.md` under Code Review Rules.
A request or reaction is not a completed review: inspect the actual bot review,
triage findings and request re-review after substantive fixes. Codex review
complements tests and maintainer judgment; it does not certify correctness or
replace the existing CI gate. Do not claim it ran if setup/authorization is missing.

## Issues and triage

Use the bug or feature/design forms. Issue titles should identify the problem,
using `bug: ...` or `feat: ...`; they need not be commit messages.

Bugs need Ariadne and Hermes revisions, Python/OS/interface, a minimal synthetic
reproduction, expected/actual behavior and sanitized evidence. Distinguish an
observed failure from a hypothesis. Proposals need the use case, alternatives,
bounded acceptance criteria and compatibility/data implications. Do not report an
unimplemented roadmap feature as a regression.

New reports start with `needs-triage`. Maintainers classify with `bug`,
`enhancement`, `documentation`, `testing`, `ci`, `security`, `breaking-change` or
`dependencies`; `blocked` means an explicit dependency prevents progress. Priority
labels (`priority: high`, `priority: normal`, `priority: low`) are maintainer-set,
not a response-time promise. Issues are closed as completed only when their
acceptance criteria are actually met; otherwise use a reason such as duplicate,
not planned, or superseded. Do not silently bury follow-up work in PR comments.

## Releases and license

Contributions are under [Apache-2.0](LICENSE), without a separate CLA. Retain any
upstream copyright, license and NOTICE obligations when code is actually ported.
The offline pure tree/view module is ported from the pinned UniiChat source;
retain its notices in source and distribution artifacts (see NOTICE).

Keep `main` installable as an experimental development milestone. Version tags
use `vMAJOR.MINOR.PATCH`; unstable public releases must be marked as prereleases
and explain current limitations, compatibility, migrations and rollback. Publishing
this repository does not publish a PyPI package or create a stable release.

## Where these conventions come from

See [repository conventions and references](docs/repository-conventions.md).
We borrow upstream principles, not their project-specific commands or contribution
restrictions. The actual PR template is the single maintained submission checklist.
