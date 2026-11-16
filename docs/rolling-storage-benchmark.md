# Delta storage benchmark evidence

Measured runtime revision: `213cd770fbc30093cd4e805a30af49d8d9f35711`. Campaign completed (result-file UTC timestamp): `2026-11-16T20:29:30.740497+00:00`.

**The cold-open acceptance gate fails.** Delta storage reduces persisted payloads and warm-open latency, but does not make first-open validation usable at the 100,000-event cap. Keep #19/#26 acceptance open; this is not activation approval.

## Method and boundaries

- Linux aarch64, two available CPUs, 11.65 GiB RAM; Python 3.14.9.
- Committed `scripts/bench_rolling_open.py`, synthetic native-shaped Codex exchanges, no provider/network calls. Default 100,000-event prefix cap; 64 generations. Legacy reader/writer baseline `ddc433f`; new reader `213cd77`.
- Three untraced repetitions per legacy/native/migrated fixture at five event counts, plus depth variation: **51 completed cold/warm measurement pairs across 17 groups**. Each group has exactly three repetitions.
- Sequential, after tests/builds stopped. Cold = empty verification cache, **not** cold OS page cache; metadata/statistics/fingerprint reads can warm the filesystem cache.
- Fixtures built once; native fixtures reused where encoding is unchanged and validated by the final reader. Legacy originals never migrated in place: migration uses separate copies.
- Exact logical SQL/artifact fingerprints unchanged before/after every measured open; source checks equal event count on both cold and warm opens. All five migrations preserved the canonical archive and mapped 64 targets/jobs/generations.
- Payload bytes below are serialized target + generation bytes only, excluding the canonical archive, SQLite/page overhead and retained backups. Active migrated storage equals native storage; conversion does not reclaim the retained legacy snapshot/root.

## Open latency, seconds

Each cell is median **[minimum–maximum]** over three repetitions.

| Events | Legacy cold | Native delta cold | Migrated delta cold | Legacy warm | Native warm | Migrated warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.672 [0.667–0.707] | 1.713 [1.695–1.751] | 1.683 [1.654–1.981] | 0.173 [0.171–0.180] | 0.179 [0.176–0.182] | 0.184 [0.184–0.205] |
| 4,096 | 8.013 [7.468–8.999] | 6.903 [6.691–6.991] | 6.873 [6.763–6.912] | 0.527 [0.481–0.851] | 0.264 [0.251–0.269] | 0.266 [0.264–0.269] |
| 16,384 | 29.105 [29.032–29.447] | 23.178 [22.980–23.490] | 22.813 [22.233–22.821] | 1.361 [1.305–1.362] | 0.520 [0.490–0.539] | 0.516 [0.515–0.559] |
| 65,536 | 117.630 [117.371–123.378] | 87.697 [86.784–89.443] | 88.610 [88.058–90.540] | 4.869 [4.799–5.047] | 1.581 [1.554–1.641] | 1.579 [1.573–1.612] |
| 100,000 | 180.953 [180.313–181.160] | 134.498 [134.030–134.823] | 135.463 [135.322–136.362] | 7.621 [7.444–8.092] | 2.443 [2.370–2.510] | 2.457 [2.447–2.674] |

At the cap, native cold median improves by **25.67%**, but remains **134.498 s**, far above the proposed ≤5 s threshold. Native warm improves by **67.94%** to **2.443 s**. Migrated behavior is comparable, not a separate performance solution.

## Stored payloads and references

MB here is decimal (1,000,000 bytes). References count stored event-ID entries across targets and generations, not unique IDs or token usage.

| Events | Legacy payload MB | Delta payload MB (native/migrated) | Legacy event-ID references | Delta references |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.395 | 0.185 | 16,640 | 512 |
| 4,096 | 21.040 | 1.411 | 266,240 | 8,192 |
| 16,384 | 84.388 | 5.354 | 1,064,960 | 32,768 |
| 65,536 | 340.406 | 21.235 | 4,259,840 | 131,072 |
| 100,000 | 520.641 | 32.388 | 6,503,840 | 200,000 |

At the cap this is **93.78% less payload**, or **16.08× smaller**. Every new-format fixture stores exactly twice its event count in event-ID references across targets and generations.

## Migration cost and retained disk

Single offline conversions, not three-run medians. Disk peak sampled once per second; short-lived peaks may be missed. Values include archive, retained legacy root/snapshot and new storage, not just active payload.

| Events | Conversion seconds | Final bytes | Sampled peak bytes |
| ---: | ---: | ---: | ---: |
| 256 | 5.383 | 3,840,871 | 3,840,871 |
| 4,096 | 23.151 | 50,004,674 | 50,004,674 |
| 16,384 | 82.201 | 198,681,429 | 198,681,429 |
| 65,536 | 313.039 | 799,291,774 | 799,291,774 |
| 100,000 | 511.499 | 1,221,798,915 | 1,221,798,915 |

## Generation depth and residual work

Fixed 4,096 events, new-format storage; three repetitions each.

| Generations | Cold median seconds | Warm median seconds | Stored payload bytes |
| ---: | ---: | ---: | ---: |
| 4 | 5.277 | 0.083 | 1,317,229 |
| 16 | 5.449 | 0.120 | 1,333,451 |
| 64 | 6.903 | 0.264 | 1,410,893 |

Cold ancestor artifact reads at 64 generations are **2,207** for new storage versus **191** on warm opens; reducing stored references does not eliminate repeated ancestor verification. A preliminary 16,384-event profile also found substantial strict-serialization/canonicalization cost and duplicated generation/target source work. These are measured follow-up candidates, not permission to weaken integrity or silently broaden this storage change.

## Memory caveats

The raw JSON records preserve `maxrss_mib`, but matrix child peaks can inherit the launcher’s pre-exec footprint after it inspects large fixtures. For example, a clean-launch supplemental 4-generation/4,096-event run reported 49.6 MiB versus 570.5 MiB in the matrix. Do not present the raw small/depth values as isolated plugin memory requirements.

Supplemental clean-launch cap runs reported **1,705.2 MiB legacy** and **418.3 MiB native** peak RSS. These are separate single-run memory checks, not the timing medians above; their cold times were 184.044 s and 137.625 s respectively. The same native/migrated encoding does not imply that a separate migrated clean-launch RSS measurement was performed.

The harness retained-cache estimate at the cap was 135.3 MiB legacy and 103.8 MiB native/migrated in all three repetitions. This is an object-size estimate, distinct from RSS, peak allocation and a provider/token budget.

## Verification and remaining limits

- Runtime `213cd77`: canonical pinned-Hermes suite **3,748 passed, 2 skipped**; optional zstandard and opt-in native tracer skipped. Installed-wheel native and actual old-wheel→CLI migration→new-plugin restart traces independently pass.
- Hosted CI43936179596 passed on the measured runtime head; Codex reviewed it with no findings. Five reproduced CodeRabbit findings fixed and threads resolved; ten targeted guard mutants killed.
- This campaign certifies neither model recall/summary accuracy nor complete provider budgets, real-provider end-to-end latency, production load or daily-use readiness. Cap-sized select/navigation/completion latency is not established by an open benchmark.

Machine-readable medians/ranges, I/O counters, fingerprints and conversion records: [rolling-storage-benchmark-213cd77.json](rolling-storage-benchmark-213cd77.json).
