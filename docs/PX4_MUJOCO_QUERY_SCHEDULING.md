# PX4–MuJoCo Phase2D — Incremental diagnostics and query scheduling

Branch `feat/px4-mujoco-query-scheduling-fix`; base
`26f3df6ed1417c9c9b90696809a2c80d99a43ff5`.
Completed offline verification date:2026-10-12 (implementation began2026-10-11).
**Offline only. Zero PX4 processes, zero MuJoCo episodes, zero live arm/yaw commands.**
No controller/physics/motor/magnetic/PX4 allocation changes, no training.

## Evidence and root-cause boundary

Confirmed from immutable Phase2C raw summary:19 queries,265803 console
characters; last start wall15.435843s +.65s =16.085843s service deadline.
At the final recorded check, actuator age.002078s and actuator deadline
16.333173s. Recorded monitor fault is `safety query service stopped`, not
proof of an actuator outage. Last5 wall intervals increase from.53678 to
.64164s although queries are scheduled at.5 simulation seconds.

Confirmed source mechanisms: repeated `''.join(console)` creates historical
copies; `allocation_status` strips ANSI and scans all historical rows at each
250Hz check; listener suffixes are reparsed on every read; safety queries are
only dispatched after physical stepping, using simulation progress, not in
the actuator receive service. A wall deadline can therefore expire even with
live actuator traffic. These are targeted defects, not proof they are the
unique historical CPU/I/O cause. No per-call CPU attribution, PTY byte-arrival
timestamps or host scheduler trace was recorded. That causal attribution
remains **Unresolved**. No synthetic replay is described as live flight.

## New data flow and clocks

PTY new chunks → bounded `StreamingDiagnostics` → complete primary-topic
events → unchanged `TelemetryFreshness` publication/sample-time validation.
The allocation protocol checks a bounded host-received row snapshot, not
historical text. `ConsoleArchive` preserves raw console bytes on disk;
only the latest65536characters are accessible on the control hot path.
Final historical JSON materialization occurs after control/cleanup, not at
250Hz. No important log/health monitoring is disabled.

| Time | Meaning / use |
|---|---|
| PX4 publication `timestamp` | Original source lease; never rewritten on receipt |
| PX4 `timestamp_sample` | Finite integer sample time≤publication; cannot roll backward |
| Simulation `clock_usec` | PX4 lockstep clock, starts1e6; source-age checks |
| Host monotonic offset | Query/receipt/communication absolute deadlines |
| Query start | Before nonblocking command dispatch; includes send delay |
| Query complete | All3 complete topics (plus allocator row when required) |
| Allocator row receipt | Host query snapshot; native publication timestamp unavailable |

`BoundedQueryScheduler` maintains absolute due times in both clocks. Either
clock reaching its.5s due time requests service, at most one request. The
trigger clock advances its absolute phase past the current time; the other
clock is rebased to prevent duplicate back-to-back jobs. No catch-up queue.
Checks run at physical ticks AND between≤8ms actuator receive slices.
Query completion as well as dispatch requires an actually empty PTY, not
merely one complete chunk. Four-read budget exhaustion keeps backlog pending
and forbids new maneuvers; residual partial replies keep the original150ms
deadline. Continuous startup backlog cannot postpone the first query service
forever: its initial due time has the same650ms protection.
Old deadlines are checked before any dispatch/receipt renewal. Pending queries
forbid new maneuvers; late health cannot clear a fault. Safety jobs cannot
queue behind passive diagnostic commands. Duplicate safety listeners in
bulk diagnostic requests are removed because the scheduled service already
records those same topics. Other diagnostic logs are preserved/deferred.

The PTY is nonblocking. `BoundedConsoleWriter` retains one command batch,
handles partial writes, and fails at its original absolute deadline. TCP
sensor sends now have explicit bounded budgets and post-send deadline checks;
receive time budgets remain absolute. No forced arm or precheck bypass.
Bounded draining must observe an empty PTY before dispatch; otherwise service
is deferred without resetting any deadline. Preexisting queued bytes cannot
be attributed to a new request. Host monotonic time is resampled after raw
archival/parsing and before query completion, so slow I/O counts against the
150ms budget. Raw bytes are archived before strict incremental UTF-8 decoding;
invalid input fails closed without discarding the offending evidence.
Incomplete UTF-8 bytes cannot complete a query or authorize a maneuver;
they have an independent150ms completion deadline, including startup.

## Unchanged deadlines and failure behavior

- Query response:150ms, including output write/processing.
- Actuator watchdog:250ms; polling slices≤8ms.
- Query service and receipt watchdog:650ms.
- Source leases: LP650ms / flags1650ms / vehicle status1150ms.
- Real health loss, failsafe, GNSS fault, final heading/alignment loss,
  localization loss, rollback/future/changed duplicate publication and
  malformed/incomplete/oversize records remain fail-closed.
- A console row received outside an outstanding request cannot renew yaw
  authority. A fresh complete row is required in each query; its lease is
 650ms in either clock. Old nonzero history is not rescanned/reused.

No source timestamp exists for the native allocator text. Request serialization
and bounded response/receipt age are the available evidence, not an independent
allocation heartbeat. Console output is not an authenticated request-ID protocol;
the source/transport remains trusted loopback PX4. This limitation is disclosed,
not repaired by inventing a PX4 timestamp.

## Explicit space/work bounds

Line8192chars; record32768chars; input slices4096chars; at most4 reads per
service (16KiB); recent console65536chars; one command batch4096bytes;
8192 safety events; finite runner archive16MiB bytes. ANSI sequences may span reads.
Primary instance0 is isolated from1/2. Duplicate keys are rejected, not allowed
to overwrite faults. Missing/partial health never completes a query.
Terminal SGR styling is allowed; other terminal controls are rejected except
clear-line operations on a blank editor line or explicit shell echo. Controls
cannot fabricate a field key or indentation. The duplicate-field guard uses
the exact same whitespace grammar as the field parser. Indented and
`pxh>`-prefixed instance headers are recognized, never treated as primary
fields. Each completed record is checked before subsequent records can replace
it. Fault flags, health revocations, bad field types/nonfinite heading and
phase-applicable prerequisites latch immediately; disarmed/ascent final
alignment and in-air readiness remain distinct. Unrequested negative evidence
can stop control, but cannot renew any positive source/allocator lease.
Unrequested publications still undergo the same finite integer/nonboolean,
future/rollback/sample/changed-duplicate checks. Three separate last-observed
records preserve ordering evidence even outside requests; they never renew
the three positive source leases. Truncated safety-topic name prefixes are
rejected instead of being misclassified as unrelated topics.
Started primary records cannot disappear at a later topic header. Native
body labels, canonical fields and explicitly tagged interleaved logs are
recognized; malformed indentation/missing colons are rejected during ingestion.
Completed truncated topic boundaries are rejected; full indented headers are
normalized and still undergo the same health validation. Native type labels
before instance declarations remain supported, and secondary data never merges.
Exceeding a bound latches/stops; no silent dropping of required safety state.
Any nonterminated content, including any partial INFO allocator prefix, is
pending; only the complete idle `pxh>` prompt is exempt. Completing the three
topics does not erase the150ms deadline while another record is partial.
The archive retains the offending chunk, even when its lifetime bound stops
the run. Normal finite45s control records are also bounded by the protocol.

Parser work is O(new bytes + completed bounded record), retained parsing state
O(line bound + record bound), independent of historical log length. Archive
storage grows on disk within the explicit finite-run cap. No hard-real-time
guarantee: Linux scheduling/slow disk can still cause a safe timeout.

## Offline measurements

Report contains5 independent batches×100 calls per size with mean/P95/P99/max
and all batch summaries. Size fixtures repeat/truncate the original console;
they are not extra measured flight duration. New tests/benchmarks ingest
unchanged real Phase2B/2C raw logs, whose file and console SHA256 are recorded.

| History | Old whole-history scan mean | New fixed-record parse mean |
|---|---:|---:|
|1KiB|≈15µs|≈2–3µs|
|10KiB|≈58µs|≈2–3µs|
|100KiB|≈560µs|≈2–3µs|
|265803chars|≈1.5ms|≈2–3µs|
|1MiB|≈6–7ms|≈2–3µs|

These are isolated parser workloads; not end-to-end query RTT or complete
250Hz safety-check costs. Full batch parsing with parser construction can be
slower than a single old whole-batch parse; repeated suffix parsing costs more.
The whole-response query comparison uses the retained legacy parsing path
with the current shared validator, not a frozen historical validator CPU cost.
Only the retained allocation-history scan is byte-identical historical code.
The principal improvement is bounded critical-path work, not universal speedup.
Per-workload tracemalloc peaks, parser high-water mark, and RSS are in JSON.
Constructed12s timing at.8simulation/wall speed yields25 queries, .5s wall
intervals and no safe timeout; this is NOT a reconstructed original PTY timeline.

Constructed4ms causal checks detect missing query reply at152ms, stopped
actuator with stalled simulation at252ms, and stopped query service or duplicate
publication at652ms. All are2ms after the unchanged respective deadlines;
received failsafe latches in the same call. Late healthy data never clears
these faults. These are deterministic offline fixtures, not actual PX4 fault
onset/detection measurements. A host scheduler pause or blocked disk write
can delay the next check; no bounded hard-real-time detection claim is made.

## Verification / acceptance

Original150 tests retained unchanged. Author final scope:289 PASS /0skip
(7.91s), adding139 tests. Tests cover streaming boundaries/ANSI,
primary isolation, source leases, fault latching, delayed/written responses,
queues, slow/stalled clocks, bad timestamps, duplicate fields, memory bounds,
old raw console equivalence and exact stopping arithmetic. Every confirmed
Important/Critical finding was reproduced RED before fixing and prompted a
fresh independent review. Per-round findings are retained in the JSON report;
several share parser grammar/lifecycle mechanisms, not independent flight causes.
One interrupted review produced no final answer and was never accepted.

Final fresh read-only review (gpt-6-astra/high): **0Critical/0Important/0Minor,
offline PASS**. Reviewer independently ran289 tests (0skip,6.89s), plus144
actual `ConsoleService → AllocationYawProtocol.before_step` lifecycle cases
over malformed bodies/boundaries/chunk widths. All latched faults and retained
`step_time=None`. No runtime source changes after this final review; no live
SITL certificate. Unseen future console schemas remain unverified.

The whole-repository pytest run was also attempted and is **BLOCKED**, not
passing: three pre-existing collection errors in
`cad_check/test_inspect_stl_assembly.py`, `mujoco/scripts/test_baseline.py`,
and `mujoco/scripts/test_dynamics_v2.py` use system SciPy with venv NumPy2.2.6
(`numpy.dtype size changed`). No dependency install or unrelated fix was made.
The complete relevant bridge suite passes; this is not a claim of global
CAD/RL test certification.

Raw Phase2B/2C summaries and ULogs:8 identities reverified, unchanged. Twelve
non-monitor artifacts from the Phase2C preservation manifest remain hash-equal,
including physical XML/rotors/adapters/BC/v3 models and old reports. Both
original and isolated PX4 source/binary identities remain unchanged. The87
historical original-worktree untracked files remain present; no cleanup.

Resources: one offline analysis worker, zero live simulator/PX4 runner instances,
no training/PX4 build/install/swap change. Final analyzer peak RSS433184768bytes
(~413MiB); largest recorded earlier analyzer peak537358336bytes (~512MiB).
Largest recorded `/usr/bin/time -v` verification MaxRSS1188416KiB (~1.133GiB);
final289-test invocation1045868KiB (~.997GiB), including native offline allocator
test compiler/subprocess accounting. These are per-command high-water marks,
not an instantaneous combined process-tree or kernel-wide OOM certificate.
No OOM observed in owned jobs; system available memory remained above the
1.5GiB reserve at recorded checks. Finalization/imports are outside flight.

| Acceptance | Status |
|---|---|
| Historical stop deadline reconstruction | PASS |
| Unique historical timeout cause | BLOCKED — attribution signals missing |
| Incremental parser and explicit space bounds | PASS (offline) |
| No threshold relaxation / offline fault regressions | PASS (offline) |
| Fault detection delay / scheduling | PASS (constructed fixtures only) |
| Independent final review | PASS (offline only,0Critical/Important) |
| Unsafe admission in tested malformed input classes | PASS (no release in regression cases) |
| Whole-repository test collection | BLOCKED — pre-existing NumPy/SciPy ABI |
| Real PX4 response RTT meets150ms | NOT TESTED |
| New runtime unarmed/armed operation | NOT TESTED |
| Full yaw response / Phase3 admission | BLOCKED |

The runtime source changed. A historical disarmed proof is explicitly rejected
for this new runtime: proof must match the hashes of the streaming runner,
parser, scheduler, monitor, transport and protocol modules. Offline PASS cannot
authorize reusing the old smoke as live validation.

Reproduce offline only:

```bash
cd /home/keyousuke/MineUAV-Learning-px4-sitl
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.analyze_query_scheduling
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/keyousuke/MineUAV-Learning/.venv/bin/python -c 'import sys;sys.path.append("/usr/lib/python3/dist-packages");import pytest;raise SystemExit(pytest.main(["mujoco/px4_bridge","-q","--tb=short"]))'
```

## Rulings / next step

- Autonomous scope replaces approval menus; cost if wrong: design revision.
- Single scoped final commit; ignored plan/review records preserved rather
  than deleted; cost: local disk.
- User-required re-review overrides the workflow's one-review limit; cost:
  reviewer time.
- Native allocator data remains a bounded serialized query snapshot, because
  PX4 prints no publication timestamp/request ID; cost: no independent
  authenticated allocator heartbeat.
- Final historical materialization is allowed only after control, within the
 16MiB cap; cost: bounded finalization RAM, not a constant-space archive.
- Declined live RTT/flight certification and unique historical CPU/I/O
  attribution remain untested/unresolved; cost: new separately authorized
  live checks and better timing evidence are still required.
- Global CAD/SciPy environment repair is outside this task; cost: only the
  complete relevant suite can be certified. No unresolved Minor is deferred
  merely to declare completion.
- Unseen future native-console formats are not assumed compatible; cost:
  new matching disarmed verification and schema review if the format changes.
- Final report/count refresh is owned and checked by the author; cost if
  wrong: misleading metadata despite a correct reviewed runtime. Source and
  raw-artifact hashes are rechecked before commit.

Only next recommendation: separately authorize a new matching **disarmed**
SITL communication check before considering any bounded yaw flight. No such
check is performed or authorized automatically here.
