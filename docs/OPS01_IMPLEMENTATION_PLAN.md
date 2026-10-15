# Research Operations V1 Implementation Plan

**Goal:** Durable, testable daily project operation with zero training authority.

**Architecture:** A standard-library Python CLI manages atomic local state and real kernel locks. Read-only inventory and a pinned dependency manifest establish identity. A bounded maintenance task exercises lifecycle/stop/recover; an explicit file allowlist controls Git synchronization. No ML/PX4 entrypoint is invoked.

**Spec:** `docs/CODEX_MASTER_SPEC.md`, MASTER_PROTOCOL_V1.0.

**Global constraints:** current zero training/flight budget; preserve all history and87 main-worktree untracked files; single worker; MemAvailable>=1536MiB; disk>=2GiB; no force push/main changes; no installs/system services.

## Review focus

1. A live writer with stale PID metadata must still be excluded by flock.
2. SIGKILL must leave INTERRUPTED evidence, not a safe-stop claim.
3. Invalid/corrupt state, hash or checkpoint must prevent start, never reset budgets.
4. Partial push or remote movement must retain local commit and become SYNC_PENDING.
5. Public status files and edited local authorization values must not grant optimizer execution.

## Task 1: State, identity and environment

Files: `research_ops/storage.py`, `research_ops/identity.py`, `research_ops/environment.py`, `research_ops/environment_lock.json`; tests in `tests/research_ops/`.

- [ ] Write failing tests for atomic writes/corruption, state transitions, flock exclusion and zero-budget rejection; observe RED.
- [ ] Implement strict JSON, bounded records, atomic fsync/replace, explicit state transitions and real flock.
- [ ] Inventory historical tracked identities, protected raw evidence, untracked names/hashes, source and dependency versions. Restore pinned upstream into a durable ignored directory without changing history.
- [ ] Run targeted GREEN tests and actual read-only doctor.

## Task 2: Lifecycle and command entry

Files: `researchctl`, `research_ops/cli.py`, `research_ops/lifecycle.py`.

- [ ] RED tests: initialization, real concurrent processes, bounded maintenance start, stop request, killed process recover, bad identity, missing dependencies and attempt no-overwrite.
- [ ] Implement status/doctor/start/request-stop/recover/verify. Start supports only registered L1 maintenance, no generic command or training adapter. Recovery never resumes ML.
- [ ] Verify process boot/start identity and lock independently; reject old attempts. Persist true step evidence, checkpoint class and interruption status.

## Task 3: Documents and Git synchronization

Files: `research_ops/sync.py`, `research_ops/documents.py`, public docs and JSON report.

- [ ] RED tests using real temporary Git repositories: allowlisted commit/push, dirty unrelated files preserved, secret rejection, failed push, remote SHA verification, SYNC_PENDING retry.
- [ ] Implement reproducible status/phase/daily documents, exact allowlist and branch checks. Do not stage arbitrary user files. Record successful remote SHA only after verification.
- [ ] End-to-end normal shutdown and SIGKILL recovery tests; record no training/PX4 execution.

## Task 4: Delivery

- [ ] Run the no-update historical suite plus OPS tests, record counts/exclusions and resource peaks separately.
- [ ] Independent read-only review; resolve Important/Critical with regressions.
- [ ] Finish all required docs and machine report; commit explicit files, push feature branch, verify remote SHA; local state records remote success.

## Decisions and progress ledger

- Ruling: use actual984a1fd successor as OPS base, preserving the already-provided L9S history; do not repeat RL-01. User77e921f baseline remains recorded as the historical reference.
- Ruling: OPS starts only a bounded read-only audit task. Training authorization installation and arbitrary experiment execution are UNSUPPORTED until a later explicitly authorized adapter/protocol exists.
- Ruling: current directory is already an isolated worktree; keep it and its local-only evidence.
- Tasks1–3 implemented; final OPS suite49 PASS,0 failures/errors/skips. Real temporary Git/process fixtures verify exclusivity, safe stop, SIGKILL recovery, push failure/retry and source-bound readiness. Six Important initial independent findings reproduced RED and fixed GREEN; final closure is SELF_REVIEW, not a fabricated independent final certification.
- Actual workspace init/doctor/status/start/recover/verify passed; durable upstream restored without packages or history changes. Runtime source frozen as `bd02c8049b806056c47e553c92d32886d0406fc3c777300f1a6b1e8beedf899e`; protocol `59964b4d2bba385308e62ff0b9f9d8d0818e53ba767aaf398ec20fa95007a5a8`.
- Related inherited regression:885 PASS,15 explicit deselections; earlier mixed-root run812 PASS/64 FAIL/9 setup errors preserved. Root cause is old hardcoded/tmp source paths; no historical code changed. New OPS actual runtime uses durable source successfully. Total934 selected unique passing cases in two complete runs, not a single full-repository test.
- Ruling: frozen old tests are retained, so full legacy regression uses its consistent historical path; permanent-path test portability is documented UNSUPPORTED pending a separate L1 migration. No identity check was weakened.
- Ruling: final fix verification is author SELF_REVIEW after independent initial NO_GO and source-bound regressions; do not claim an independent final code GO that was not performed.
- Task4 source/docs/report complete; final delivery action is `./researchctl request-stop` and verified sync. Commit/remote completion lives in `.research_local/research_state.json` and Git, rather than a self-referential pre-commit SHA in this plan.
