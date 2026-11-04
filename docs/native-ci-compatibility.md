# Native CI dependency compatibility

CI checks out the inspected public Hermes source revision
[`c627ec84664d43e0b18b95d9d86d8963747d56c7`](https://github.com/NousResearch/hermes-agent/tree/c627ec84664d43e0b18b95d9d86d8963747d56c7).
Its [`uv.lock`](https://github.com/NousResearch/hermes-agent/blob/c627ec84664d43e0b18b95d9d86d8963747d56c7/uv.lock)
contains **OpenAI 2.54.0** and **httpx 0.28.1**; its Python 3.14 project
requirements also pin OpenAI 2.54.0. That is not the native compactor's
source-validated transport pair, **OpenAI 2.24.0 / httpx 0.28.1**. The exact-version
guard therefore correctly rejects the unmodified CI host rather than claiming
support for an uninspected SDK.

The workflow keeps the host source pin and PM build unchanged. At this revision,
`pm.build_env` defaults to a frozen build (`frozen=not args.resolve`), ultimately
using `uv sync --frozen --all-packages`; `--ci` does not replace SDK pins. Before
any dependency override, CI runs the baseline Engine, host-contract, restart and
install test files against that pristine PM dependency set and reports its actual
imported SDK versions.

Only then does CI explicitly install `openai==2.24.0 httpx==0.28.1` into the
**disposable `$RUNNER_TEMP/hermes-python` test environment**. Pip runs from the
separate development venv and targets the host interpreter with `--python`, so
the PM-created venv need not include pip. This is an intentional, logged
**non-pristine dependency override**, not evidence that the unmodified PM lock
supports native compaction. The workflow checks both imported `__version__`
values and installed distribution metadata before running **the entire test
suite**, including all native and loopback transport tests, without test filters
or SDK-version spoofing. `scripts/test.sh` exposes that host's site-packages to
the development pytest interpreter through `PYTHONPATH`.

No installed/managed Hermes runtime, user configuration, production dependency
lock, host source, or SDK support guard is modified. These checks use isolated
homes and loopback provider fixtures, not real models or private credentials.
A hosted run of the updated workflow is still required to verify the complete
PM build, pristine baseline tests, real package override, version assertions,
and full-suite execution together.
