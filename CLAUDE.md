# colab-link

Goal: run Python jobs on rented Colab GPUs on demand: start a session → set up the environment (cached) → run → return results → stop the VM. The main uses are LLM processing and OCR with a multimodal LLM, but **it is not limited to batch jobs**; any Python job script works.
Phases: ① Colab feasibility (done) → ② Drive cache (done) → ③ **CLI (current; the HTTP API comes later)** → ④ HTTP API.

**Keep this file updated**: whenever you verify a new fact, hit a pitfall, or make a design decision, write it here so the next session can pick it up.

## Working rules (from the user)
- Reply to the user in Traditional Chinese.
- **Claude usage check**: at the end of every stage, run `python3 tools/claude_usage.py` (exit code 2 = over 90%). If over 90%: first finish writing the current file, make sure no Colab session is still running (`colab sessions`, stop any leftover to avoid wasting CU), then schedule a resume after the reset time (`resume_at`) and continue from there. Don't stall.
- Models: use newer Qwen versions, pulled from HF for now. Moving them to Google Drive is to be discussed later.

## Budget
- **200 CU per month**; the user wants to use as much of it as possible. Check the balance with `colab usage`.
- Measured rates (2026-10-14, CU per hour):

| GPU | VRAM | CPU / RAM | CU/h | Hours from 200 CU |
|---|---|---|---|---|
| T4 | 15 GB | 2 / 12 GB | 1.07 | ~187 |
| L4 | 23 GB | 12 / 52 GB | 1.54 | ~130 |
| A100-40GB | 40 GB | 12 / 83 GB | 5.30 | ~38 |
| G4 (RTX PRO 6000 Blackwell) | 96 GB | 48 / 176 GB | 8.90 | ~22 |
| H100 | — | — | not available on this account | — |

- Default to **L4** (best value; bf16; fits 7–8B multimodal models). Use G4 or A100 only for large models.
- Provisioning is fast (about 6–11 s to READY). Each new VM is clean, so models are downloaded again every time (T4 about 63 MB/s; on L4, 9 GB took 30 s, about 300 MB/s).
- **Concurrency limit (measured 2026-10-14)**: at most **2 sessions per GPU type**, and at most **5 sessions in total** across all types including CPU. When refused, the error is `Allocation refused (precondition failed)`. T4, L4 and A100 can run at the same time.

## Inference engine: vLLM (decided)
- Reason: billing is per hour, so throughput is what matters. vLLM's continuous batching suits batch jobs; it reads HF safetensors directly, supports FP8/AWQ/GPTQ, and has mature Qwen-VL multimodal support. Installing it on the VM takes about 37 s.
- GGUF/llama.cpp is better for squeezing a big model into a small GPU. Here we can switch to a larger GPU or use FP8 instead, so it is not the main path. If needed, benchmark both with the same set of tasks.
- Qwen3.5+ uses thinking mode by default. `jobs/vlm_batch.py` turns it off by default (`ENABLE_THINKING=1` turns it back on).
- On L4/G4 you can set `QUANTIZATION=fp8` for on-the-fly FP8 (L4 is Ada, G4 is Blackwell; both have native FP8).

## Newest Qwen models (checked on HF 2026-10-14)
| Model | Weights | Fits on |
|---|---|---|
| Qwen3.8-27B-FP8 (2026-08, multimodal) | 31 GB | A100-40GB (tight) / G4 |
| Qwen3.8-27B | 56 GB | G4 only |
| Qwen3.8-Flash-Next (180B) | 185–360 GB | does not fit |
| Qwen3.6-35B-A3B-FP8 (MoE) | 37.5 GB | G4 |
| Qwen3.5-9B / 4B | 19.3 / 9.3 GB | L4 |

## Colab CLI essentials (`google-colab-cli` 0.7.4)
- **PYTHONPATH conflict**: Hermes injects a Python 3.14 `PYTHONPATH`, which crashes the CLI on `pydantic_core`. `~/.local/bin/colab` was changed into a wrapper that runs `unset PYTHONPATH` first. After `uv tool upgrade` it may revert to a symlink; if so, rebuild the wrapper. `colab_link/session.py` also removes the variable when it calls the CLI.
  - Source (per Jarvis, 2026-10-14): the tmux server was started by the Hermes terminal tool, and Hermes `pm/environments.py` `activate_dependencies` exports PYTHONPATH (a leftover from the old environment `04c7bbc4…`). Jarvis has removed PYTHONPATH from the tmux global environment and the `claude-rc-colab-link` session, and restarted Claude with `env -u PYTHONPATH`. Verified: the Claude process and its child shells no longer have PYTHONPATH, and running the uv tool's `colab` directly (bypassing the wrapper) works. The underlying Hermes issue is not fixed yet (Hermes terminal child processes may still carry it), so **keep the wrapper for now and do not change `~/.bashrc`**.
- **Auth**: oauth2. The token is at `~/.config/colab-cli/token.json` (account mirror770109@gmail.com). On expiry the CLI prompts for an authorization code. In a non-interactive session: run it in the background with a FIFO as stdin, hand the URL to the user, then write the code they send back into the FIFO. **The user wants the URL whenever a login is needed.**
- **`exec --timeout` only bounds how long the CLI waits** (default 30 s). On timeout the CLI exits with an error, but **the remote kernel keeps running**. So long jobs must not run through `exec` directly; launch them detached on the VM and poll (that is how `runner.py` works).
- Kernel state persists across `exec` calls in the same session. Default cwd is `/content`.
- `--config <file>` isolates the local session state. The runner gives every job its own state file so jobs can run in parallel.
- Always `stop` when done; the runner stops the VM in a `finally` block. If a leftover session is suspected, check `colab sessions`.
- Do not run `repl`, `console`, `auth` or `drivemount` interactively; they need a TTY.
- VM image (2026-10): Python 3.13, torch 2.11 + cu130, transformers 5.18; **vLLM is not preinstalled**. About 207 GB of disk.

## Google Drive (2026-10-14)
- **`colab drivemount` (DriveFS) cannot be automated**: it uses Colab's credentials propagation (`dfs_ephemeral`). The authorization URL carries the VM's `endpoint`, so **every new VM needs a human to grant access once**. It also needs the same CLI process to stay open and wait for Enter after the grant (it reads `/dev/tty`); we have no TTY, so the process just fails and the next attempt generates new temporary credentials, making the earlier grant useless. Tried twice; both failed. ⇒ Do not use mounting.
- **Use the Drive REST API directly instead**: `tools/drive_token.py` (run with the CLI venv's python) obtains an access token from the CLI's own oauth2 login. Its scopes include `drive.file`, which can only access files created by this OAuth client — enough for a model/environment cache, with no further authorization needed.
  - That token belongs to the Google Cloud SDK client, so calling the Drive API **requires the header `x-goog-user-project: <GCP project>`**, and that project must have the Drive API enabled.
  - **Using `flarex-439501`** (the user's choice; the Drive API was enabled on 2026-10-14 through the serviceusage API). Usage: `runner --drive flarex-439501`.
  - The token is sent to the VM by uploading a file (do not put it in exec code, because the CLI records executed code in `~/.config/colab-cli/history`).
- **Drive location**: the `colab-link-cache` folder at the root of My Drive (https://drive.google.com/drive/folders/111AITU93plTyk1L8aVoZEa0kZHRSrIjh). It is found **by name**, so it can be moved but must not be renamed. Cache files count toward the user's Drive quota.
- `colab_link/drive_vm.py` (runs on the VM, stdlib only): `ls` / `put` (resumable, 64 MB chunks, replaces a file of the same name) / `get` (16 threads × 64 MB ranged parallel download).
- **Measured speeds (L4, 2026-10-14)**: Drive download **~256 MB/s** (8.3 GB in 32 s), about the same as HF (~300 MB/s); upload only **~31 MB/s** (8.3 GB in 269 s).
  - ⇒ **Keep pulling model weights from HF**; putting them on Drive is no faster.
- **venv cache**: `venv-vllm0.31.0-py313-torch2.11.0-cu130.tar` (8.3 GB, uncompressed; the VM has no zstd), built by `jobs/build_env_cache.py`. Path is `/content/venv` (`--system-site-packages`).
  - Restoring from Drive takes about **55 s** (32 s download + 23 s extract). Installing straight from PyPI ranges from 34 s to 4 min, so the cache's main benefit is **stability**, not saved time.
  - Run it with `--python /content/venv/bin/python`. The runner automatically puts that `bin/` on PATH (otherwise vLLM can't find `ninja`).
  - The cache name includes the image's Python and torch versions; rebuild it when Colab updates its image.
- **Engine init breakdown (L4, Qwen3.5-9B fp8, 344 s total)**: image encoder profiling ~56 s, torch.compile 62 s (cache: `~/.cache/vllm`), initial profiling/warmup 88 s (part of it is Triton kernel compilation, cache: `~/.triton`), CUDA graph capture 41 s, **flashinfer sampling JIT ~100 s** (cache: `~/.cache/flashinfer`). Before that: weight download 55 s and model load ~70 s.
  - ⇒ The JIT/compile caches are what save real time. **Cache names must include the model, quantization and GPU** (e.g. `jitcache-qwen35-9b-fp8-L4.tar`), because the compiled results depend on all of them.
  - The `.pyc` files in the venv cache barely helped (load_s 523 → 498 s).
- **Cache results (L4, Qwen3.5-9B fp8, 200 tasks, 2026-10-14)**:

| Variant | Total time | CU | load_s | init engine | Notes |
|---|---|---|---|---|---|
| Install vLLM from PyPI (baseline) | 668 s | 0.29 | 523 s | 353 s | |
| Drive venv | 744 s | 0.32 | 498 s | 344 s | provisioning happened to take 110 s |
| Drive venv + vLLM compile cache | 596 s | 0.25 | 430 s | 288 s | torch.compile 62 → 2.3 s |
| **Drive venv + all JIT caches** | **483 s** | **0.21** | **304 s** | **176 s** | flashinfer JIT skipped; 100/100 correct |

  - **Saves about 185 s (28%) and about 0.08 CU per run.** Restoring (venv 8.3 GB + JIT cache 90 MB) takes about 75 s.
  - `jitcache-<model>-<quant>-<gpu>.tar` = `~/.cache/vllm` + `~/.cache/flashinfer` + `~/.triton` (extract into `/root`). Run once without a cache and use `--post` to save it.
  - Still not cacheable: HF weight download 54 s, initial profiling/warmup 89 s (the Triton cache does not shorten it; it is mainly image-encoder profiling and dummy runs), CUDA graph capture 40 s.
  - Next optimizations to try: lower the multimodal profiling cost (vLLM's `skip_mm_profiling`, or limit image size); drop packages from the venv that already match the system's to shrink the 8.3 GB; make cache restore and save a built-in runner option instead of hand-written `--setup`/`--post`.

## Layout
- `bin/colab-link` → `colab_link/cli.py`: user entry point (profiles / run / cache / sessions / stop-all / usage).
- `profiles.toml` + `colab_link/config.py`: profiles, venv definitions, Drive settings.
- `colab_link/session.py`: `ColabSession`, a thin wrapper around the CLI (start/wait_ready/exec/upload/download/stop/forget). Each job has its own state file under `~/.cache/colab-link/state/`; it is deleted once the session ends. `drive_token()` gets a token from the CLI login.
- `colab_link/runner.py`: `run_job()` runs one job end to end: new → upload job.py, in.tar.gz and the VM helpers → detached launch (setup → pip → job → post; exit code written to `exitcode`) → poll (Drive token refreshed every 40 min) → download out.tar.gz → stop. Writes `run.json`.
- `colab_link/nats_runner.py`: **default transport**. `run_job_nats()` = tar input → Drive `colab-link-jobs/job-<id>-in.tar.gz` → new VM → upload the agent and secrets → launch `worker.py` → wait for `.ready` → request `.job` → receive `.log/.progress/.heartbeat` → `.result` → fetch output from Drive → stop VM (always) → delete the job files on Drive (unless `--keep-files`). No heartbeat for 90 s ⇒ VM presumed dead and stopped.
- `colab_link/worker.py` (on the VM, edge worker): see its docstring for the subject list. One job at a time; `.ctl` accepts `drive_token` (the token is refreshed every 40 min over NATS) and `shutdown`.
- `colab_link/runner.py`: the old exec-polling transport (`run --transport exec`), kept as a fallback.
- `colab_link/vm_env.py` (on the VM): restores or builds the venv and JIT caches, and uploads anything new after the job.
- `colab_link/drive_vm.py` (VM or local): Drive API ls/put/get/rm.
- `jobs/vlm_batch.py`: vLLM batch over `tasks.jsonl` (text or text + image) → `results.jsonl` + `metrics.json`.
- `jobs/build_env_cache.py`: early venv cache experiment (replaced by vm_env.py).
- `examples/`: `make_bench.py` (100 OCR + 100 JSON tasks), `make_ocr_demo.py`, `score.py`. Generate data with `env -u PYTHONPATH uv run --no-project --with pillow ...`.
- `probe/`: `probe_env.py` (VM hardware, packages, download speed), `stability.py`, `concurrency.py`.
- `tools/claude_usage.py` (Claude usage), `tools/drive_token.py` (run with the CLI venv's python).
- Version control: git, `main` branch, remote `git@github.com:flarexio/colab-link.git` (private). `results/`, `logs/` and generated test data are not committed (see `.gitignore`); write important numbers into this file.

## Running (CLI: `bin/colab-link`)
```bash
bin/colab-link profiles                                   # list profiles (profiles.toml)
bin/colab-link run qwen35-9b-l4 --input examples/bench    # output defaults to results/<profile>-<time>
bin/colab-link run qwen35-9b-l4 --input DIR --env MAX_MODEL_LEN=4096 --dry-run   # print the plan only
bin/colab-link run --gpu T4 --job my.py --input DIR --pip pandas                  # no profile: any script
bin/colab-link cache ls | cache rm NAME...                # caches on Drive
bin/colab-link sessions | stop-all | usage                # sessions on the server / stop leftover VMs / CU balance
```
- `bin/colab-link` runs `python -m colab_link` in the project's **uv venv** (`uv run --project`, with PYTHONPATH cleared). Dependencies: `pyproject.toml` (`nats-py[nkeys]`, `aiohttp`); `.venv/` is not committed.
- **Division of labor (the user's design)**: files go through **Google Drive** (users upload over HTTP to the colab-link server, which writes them to Drive); **NATS carries only control messages** (which file to process, progress, results). Never send files over NATS.
- NATS subjects: `edges.<edge>.colab-link.workers.<vm>.{ready,heartbeat,job,log,progress,result,status,ctl}` (consistent with the user's existing logic). The service endpoint planned later: `edges.<edge>.colab-link.{run,status,cancel}`.
- **Job contract**: `job.py IN_DIR OUT_DIR`. The whole `--input` directory is sent to the VM as `in/`, and everything the job writes to `out/` comes back to `--out` (along with `log.txt`, `env_state.json`, `run.json`). The job must put its work behind `if __name__ == "__main__"`.
- **Profile** = GPU + venv + default job + env. Profiles that have a venv automatically go through `vm_env.py prepare/finalize`:
  - venv cache key: `venv-<name>-<sha(packages)>-py<ver>-torch<ver>.tar` (computed on the VM, so it changes with the image).
  - JIT cache key: `jit-<profile>-<sha(gpu+env)>-<sha(venv key)>.tar`. Changing the model or parameters with `--env` gives a new key automatically.
  - If a cache is missing, it is built on the spot; after the job **succeeds**, it is uploaded to Drive (a venv upload takes about 270 s, once only). `--no-cache` turns this off.
- The low-level `python3 -m colab_link.runner` is still available (hand-written `--setup/--post/--python/--drive`).
- 2026-10-14 **NATS transport verified (CPU, jobs/hello.py)**: VM start → worker ready 44 s (including installing nats-py); progress and logs pushed with ~1 s delay; **the result arrived and the VM was stopped within 3 s of the job ending** (exec polling took at least 20 s, at worst +2 minutes); Drive job files cleaned up automatically.
- 2026-10-14 **NATS transport verified (L4, qwen35-9b-l4, 200 tasks)**: **481 s total** (the exec transport took 596 s); 100/100 correct; both caches hit; the Drive jobs folder was cleaned up automatically.
  - Breakdown: input upload 5.6 s, provisioning 11 s, agent ready 26.5 s, on-VM env (venv + JIT restore) 91 s, input 4 s, job 331 s (model load 303 s + generation 20 s), output 2.8 s + download 1.4 s.
  - The vLLM log shows many `triton_bundler ... Cubin file ... not found at /tmp/torchinductor_root/...` messages: these are **warnings** (W) and do not affect results. The torch inductor cache in `/tmp/torchinductor_root` is not in the JIT cache yet; **adding it should save more of the compile time** (to try next).
- 2026-10-14 **CLI end-to-end verified**: `run qwen35-9b-l4 --input examples/bench`. The venv key and JIT key computed on the VM both hit the cache; 100/100 correct; 596 s total.
  - **A poll exec sometimes loses its reply** (3 times in this run); the CLI waits out the whole timeout, so the runner noticed about 2 minutes after the job ended. The poll is now `timeout=20, retries=1`. **The real fix is to switch to NATS push (see below).**

## NATS (FlareX edge convention)
- The user's server: `wss://nats.flarex.io` (JetStream + TLS, **decentralized JWT**).
- **FlareX convention (follow it)**: config directory `~/.flarex/colab-link/` holds `id` (the **edge id**, a ULID; it is also the JWT user name; the edge id is shared across products on this machine, e.g. claude-runner uses the same one) and `user.creds`.
  - Subjects: **`edges.<edge id>.colab-link.>`** (same as claude-runner's `edges.<edge id>.claude-runner`; it is the same tenant's subject namespace).
  - Connection name: `Colab Link - <edge id>` (the VM side uses `Colab Link VM - <edge id>`).
  - claude-runner uses `micro.AddService` + `srv.AddGroup("edges.<id>.claude-runner")` to expose endpoints (source in `~/go/pkg/mod/github.com/flarexio/claude-runner@v1.0.6/cmd/claude-runner/main.go`).
- **Permissions of this creds**: the JWT has no pub/sub restrictions and no expiry (issued 2025-06), so it has full permissions in the account. **Sending it to a VM is fine for testing; long term, colab-link should issue each VM a short-lived JWT scoped to `edges.<id>.colab-link.workers.<vm>.>`** (requires the account signing key).
- **Never print or log the creds**; send them to the VM by uploading a file (chmod 600). Read only the JWT's public claims to inspect permissions.
- Python: `nats-py[nkeys]` (without the nkeys extra, JWT login fails) + `aiohttp` (needed for wss). Subscription callbacks must be `async def`.
- **2026-10-14 measurements** (`probe/nats_probe.py`):
  - Local: connect 0.63 s, request/reply loopback ~20 ms.
  - **Colab VM (CPU, us-east) connecting out over wss 443: works**. Connect 1.4 s; from starting the agent to ready 7.3 s (including installing nats-py). Local ↔ VM request/reply median **197 ms** / p95 198 ms. VM-pushed heartbeats arrive with ~103 ms latency.
  - ⇒ Replacing the poll exec (~5 s each, sometimes stuck for 60 s) with NATS push is entirely feasible.
- Architecture: the control plane stays on the colab CLI (start the VM, send the agent and credentials, stop); the data plane uses NATS (dispatch jobs, progress, logs, results, heartbeats). Main benefit: **warm workers** (pull jobs from a JetStream work queue continuously and only stop after N idle minutes, so the ~8-minute startup cost is paid once).
- Still to check: Colab's terms on long-lived connections to external services and background work.

## Validation log
- 2026-10-14 **End-to-end success** (L4, Qwen3-VL-4B, 5 tasks): 391 s total ≈ 0.17 CU. 3/3 OCR outputs correct (only whitespace differed); Chinese summary and JSON extraction both correct.
  - Time split: provisioning 11 s, upload 9 s, job 360 s (vLLM install ~37 s, weight download 30 s, **vLLM engine init 192 s** (including torch.compile 48 s), generation only 1.8 s).
  - ⇒ Each session has about 5 minutes of fixed overhead, so batch jobs should be large to amortize it. Possible optimizations: `ENFORCE_EAGER=1` (skips compile; lower throughput), or caching the torch.compile cache / weights (Drive).
- Pitfalls hit (already handled in code):
  - `colab upload` returns 500 when the parent directory does not exist → create it with `exec` first.
  - Just after `colab new` reports READY, the kernel proxy can still `ReadTimeout` → `wait_ready()` pings until the kernel answers; exec/upload/download retry 3 times; launch uses a marker file so it is idempotent.
- `tools/claude_usage.py`: reads the Claude 5-hour and weekly usage (from the OAuth usage endpoint).
- Benchmarks: `examples/make_bench.py` (100 OCR + 100 JSON tasks, with expected answers), `examples/score.py` for scoring.
- 2026-10-14 **Benchmarks** (`examples/bench`: 100 OCR receipts + 100 JSON extractions; results in `results/bench-*`):

| Combination | Total time | CU per run | Generation (200 tasks) | Marginal CU per 1000 tasks | OCR exact | JSON |
|---|---|---|---|---|---|---|
| L4 + Qwen3.5-9B (`QUANTIZATION=fp8`) | 668 s | **0.29** | 20.6 s (515 tok/s) | **0.044** | 100/100 | 100/100 |
| G4 + Qwen3.8-27B-FP8 | 455 s | 1.13 | **8.0 s** (1336 tok/s) | 0.099 | 99/100 | 100/100 |
| A100 + Qwen3.8-27B-FP8 (eager, len 4096, seqs 64) | 665 s | 0.98 | 64.3 s (167 tok/s) | 0.47 | 100/100 | 100/100 |

  - **Conclusions**: L4 + Qwen3.5-9B is the best value for everyday OCR and extraction. Use G4 for big batches or harder tasks with Qwen3.8-27B. **A100 is not worth it**: no native FP8 (Marlin weight-only), and it OOMs during CUDA graph capture (it needs `ENFORCE_EAGER=1`, which makes it slow).
  - **Fixed overhead dominates the cost**: vLLM install 37 s–4 min (PyPI speed varies a lot by VM), weight download (31 GB ≈ 103 s, about 300 MB/s), engine init 130–350 s. Generation itself takes only seconds. ⇒ Optimization priorities: cache the venv and weights (Drive), cache the torch.compile results, and pack each batch as large as possible.
  - CU actually spent on the whole day's tests (including failed runs and probes): about 3 CU (200 → 197.06).
- 2026-10-14 **Connection stability** (`probe/stability.py`, 2 CPU sessions, 60 minutes):
  - One exec every 30 s: **120/120 succeeded**, latency median 4.85 s / p95 6.16 s / max 15.2 s. ⇒ Each exec costs about 5 s, so a 15–30 s polling interval is reasonable.
  - A session left completely untouched for 60 minutes while a background process ran: **not reclaimed**; the heartbeat was still updating (4 s old). ⇒ A detached job can run unattended; it is fine for the local machine to disconnect mid-run and reconnect later.
  - Not yet tested: longer than 60 minutes, and the case where the kernel is idle **and** there is no background process (the idle-timeout limit).
- Pitfalls hit (already fixed):
  - **The job script must keep its work behind `if __name__ == "__main__"`**: with Qwen3.5+/3.8, vLLM starts its engine with multiprocessing spawn, which re-imports the main module, causing a `bootstrapping phase` RuntimeError.
  - Poll exec failing 10 times in a row → the runner gives up (avoids waiting out `max_hours` after the VM is reclaimed).
  - Qwen3.5-9B writes `amount` as a string like `"474.03 USD"`. If downstream needs a number, ask for it explicitly in the prompt or use guided JSON.
