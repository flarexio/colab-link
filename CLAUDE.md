# colab-link

Goal: run Python jobs on rented Colab GPUs on demand: start a session → set up the environment (cached) → run → return results → stop the VM. The main uses are LLM processing and OCR with a multimodal LLM, but **it is not limited to batch jobs**; any Python job script works.
Phases: ① Colab feasibility (done) → ② Drive cache (done) → ③ CLI (done) → ③.5 NATS edge worker + **service mode/warm workers/container (current)** → ④ HTTP API (upload to Drive + call `run`).

**Keep this file updated**: whenever you verify a new fact, hit a pitfall, or make a design decision, write it here so the next session can pick it up.
**This repo is public**: put no credentials, account emails, GCP project ids, Drive folder ids or edge ids here. Those, along with the user's working rules and local environment notes, go in `CLAUDE.local.md` (gitignored, loaded automatically).

## Budget
- Check the CU balance with `colab usage` (or `colab-link usage`).
- Measured rates (2026-10-14, CU per hour):

| GPU | VRAM | CPU / RAM | CU/h | Hours per 200 CU |
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
- **Auth**: oauth2. The token is at `~/.config/colab-cli/token.json` (account: see CLAUDE.local.md). On expiry the CLI prompts for an authorization code. In a non-interactive session: run it in the background with a FIFO as stdin, hand the URL to the user, then write the code they send back into the FIFO.
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
  - The project comes from `COLAB_LINK_DRIVE_PROJECT` (or `[drive] project` in profiles.toml). Locally it is set in the gitignored `.env`, which `bin/colab-link` loads; the value is in CLAUDE.local.md.
  - The token is sent to the VM by uploading a file (do not put it in exec code, because the CLI records executed code in `~/.config/colab-cli/history`).
- **Drive location**: the `colab-link-cache` folder at the root of My Drive. It is found **by name**, so it can be moved but must not be renamed. Cache files count toward the user's Drive quota.
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
- `colab_link/service.py`: **`colab-link serve`** (the server-side service; see the "Service mode" section). `colab_link/client.py`: submit/fetch/status for the CLI.
- `colab_link/guard.py`: reaper + leases (see "Idle-VM guard"). `colab_link/colab_api.py` (stdlib, local and on the VM): list/unassign sessions through the Colab API. `colab_link/openai_gateway.py`: local OpenAI HTTP endpoint → NATS LLM workers.
- `colab_link/jobhost.py` (on the VM): keeps a job module that defines `handle(in_dir, out_dir)` loaded, so a warm worker loads the model only once.
- `Dockerfile` / `compose.yaml`: container for the service (image `flarexio/colab-link`).
- `.github/workflows/`: **no QEMU**; each architecture builds on its own native runner (`ubuntu-24.04` / `ubuntu-24.04-arm`, free for public repos). `build.yml` (each push: compileall + image build + smoke test `profiles` and `colab --help`, per arch, not pushed); `release.yml` (pushing a `vX.Y.Z` tag: each arch pushes by digest, then a merge job runs `imagetools create` and pushes `flarexio/colab-link:{X.Y.Z,X.Y,X,latest}` to Docker Hub; secrets `REGISTRY_USERNAME`/`REGISTRY_PASSWORD`, same as claude-runner). `README.md` is the public-facing documentation.
- `colab_link/vm_env.py` (on the VM): restores or builds the venv and JIT caches, and uploads anything new after the job.
- `colab_link/drive_vm.py` (VM or local): Drive API ls/put/get/rm.
- `jobs/vlm_batch.py`: vLLM batch over `tasks.jsonl` (text or text + image) → `results.jsonl` + `metrics.json`.
- `jobs/build_env_cache.py`: early venv cache experiment (replaced by vm_env.py).
- `examples/`: `make_bench.py` (100 OCR + 100 JSON tasks), `make_ocr_demo.py`, `score.py`. Generate data with `env -u PYTHONPATH uv run --no-project --with pillow ...`.
- `probe/`: `probe_env.py` (VM hardware, packages, download speed), `stability.py`, `concurrency.py`.
- `tools/claude_usage.py` (Claude usage), `tools/drive_token.py` (run with the CLI venv's python).
- Version control: git, `main` branch, remote `git@github.com:flarexio/colab-link.git` (public since 2026-10-15). Releases: push a `vX.Y.Z` tag; v0.2.0 published amd64 + arm64. The Docker Hub secrets are org-level and were only usable once the repo was public. `results/`, `logs/` and generated test data are not committed (see `.gitignore`); write important numbers into this file.

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

## Service mode (`colab-link serve`, 2026-10-14)
- **Architecture**: a long-running service (on the user's cloud server, in a container) = NATS micro service (`micro.AddService`, same pattern as claude-runner) + a JetStream work queue + warm Colab workers.
  - Endpoints (group `edges.<edge>.colab-link`): `run` {profile|gpu, script?, env?, pip?, cache?, idle_minutes?, **in_file**} → job record; `status` {job_id} or {} (recent 50); `cancel` {job_id}; `workers`.
  - `in_file` is a tar.gz the caller has already put in `Drive:colab-link-jobs` (later the HTTP API will upload it for the user; files never go over NATS).
  - JetStream: stream `COLAB_LINK_<edge>` (WorkQueue, subjects `edges.<edge>.colab-link.queue.<pool>`, max_age 7 days); KV `colab-link-<edge>`: `jobs.<id>` (job records, written by both the service and the workers), `pools.<pool>` (pool specs, which workers read at startup).
  - **Pool** = everything that must match for a worker to take a job: GPU + script + env + venv + pip; `pool_id = <profile>-<sha8>`. Each pool has one durable pull consumer (`pool-<id>`, ack_wait 180 s, max_deliver 3; the worker sends `in_progress` every 60 s).
  - Scaling: every 5 s the service checks each pool's `num_pending`; if there aren't enough idle workers and the pool is under `max_workers` (profile, default 1), it starts a VM. Limits: 2 per GPU type, 5 in total. When capacity is full, it asks an idle worker of another pool on the same GPU type to leave. A failed start backs off for 120 s.
  - VM stop conditions: the worker is idle for `idle_minutes` (profile, default 10) → `.bye` → the service runs `colab stop`; no heartbeat for 90 s → presumed dead, stopped, and the job goes back to `queued` (JetStream redelivers it after ack_wait); the service receives SIGTERM → it **stops every VM** (so a container restart doesn't leave billing VMs behind).
  - On restart the service **adopts** workers that are still heartbeating (it recognizes its own state files in `COLAB_LINK_STATE`); 30 s after startup, any VM with a state file but no heartbeat is stopped.
  - Drive token: refreshed every 40 minutes and broadcast to all workers over `.ctl`. Job files nobody collects are deleted from Drive after 2 days.
- **Warm job contract**: if the script defines `def handle(in_dir, out_dir)` (at the start of a line), the worker keeps it loaded through `jobhost.py`, so module globals (such as a loaded LLM) survive between jobs. Otherwise each job runs `python job.py in out`. `jobs/vlm_batch.py` and `jobs/hello.py` both have `handle()`.
- CLI: `serve` / `submit PROFILE --input DIR [--wait] [--idle-minutes N]` / `jobs [ID]` / `fetch ID` / `cancel ID` / `workers`. `submit` does the Drive upload itself (it stands in for the future HTTP API).
- **Verified (CPU, `hello-cpu`)**:
  - Cold start, submit to done: 73 s (worker ready at 49 s).
  - **Warm jobs: submit to done 23 s, run in the same process** (`pid` unchanged, `calls` 2 → 3); two jobs submitted at the same time were queued and run one after the other.
  - `kill -9` the service and restart: the running worker was adopted.
  - Cancelling a running job: killed within a few seconds, exit 130, status `cancelled`.
  - After the idle timeout the workers sent bye and the service stopped the VMs, leaving 0 sessions on Colab.
- **Container**: `Dockerfile` (multi-stage, python:3.13-slim; `google-colab-cli` installed from the `server` extra into the same venv; `COLAB_CLI_PYTHON` points at it). **336 MB** (uncompressed, arm64).
  - pyarrow (~150 MB, pulled in by jupyter-kernel-client) is removed; the colab CLI doesn't use it.
  - Volumes: `~/.flarex/colab-link` (ro), `~/.config/colab-cli` (OAuth token; the CLI refreshes it in place, so it must be writable), `/data` (state).
  - Verified: submit → VM → done inside the container works; `docker stop` stopped the VM in 7 s before exiting. **Run with `--user <uid>` when testing locally** so the token files don't become root-owned.
  - The container defaults to UTC (job ids and log times are UTC); set `TZ` if needed.
  - On this machine docker requires `sudo` (jarvis is not in the docker group). The image built here is arm64. If the cloud server is amd64, build it there or use `docker buildx --platform linux/amd64`.
- **Pitfalls**: with nats-py 2.16 + server 2.11, `kv.keys(filters=["jobs.*"])` returns nothing, so the service lists all keys and filters them itself. **`pkill -f`/`pgrep -f 'colab_link serve'` also match the Bash tool's own shell** (its command line contains the same string, so the shell gets killed too, exit 144). Use `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /colab_link serve/'` to find the PID; better still, run serve in the background and have it write its own PID file (to do). Bash tool calls are capped at 10 minutes, so for long jobs submit without `--wait` and collect the result with `fetch` afterwards.
- **Verified (L4, qwen35-9b-l4, examples/bench with 200 tasks, 2026-10-14)**:

| Case | Submit to done | Model load | Generation | Correct |
|---|---|---|---|---|
| Cold start (rebuilding JIT v2) | ~10 min | 500 s | 19.8 s | 100/100 |
| **Warm (same VM, 2nd job)** | **33 s** | **0 s** | 19.6 s | 100/100 |
| Cold start (JIT v2 incl. inductor cache hit) | ~6.5 min | **219 s** (was 304 s) | — | 100/100 |

  - Cold-start breakdown: VM + worker ready 59 s, env (venv + JIT restore) 65 s, job 240 s, output 3.5 s.
  - **Adding the inductor cache cut model load from 304 s to 219 s**, about 85 s saved per cold start.
  - Total CU spent on these tests (including CPU tests): about 0.5 (195.48 → 194.98).
- The JIT cache now also includes the torch inductor cache (`TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor`); the key gained `-v2`, so the first run rebuilds it.

## Idle-VM guard (2026-10-17, the user's requirement: no Colab session may sit billing)
Must hold even if the service crashes, this machine goes down, or **Claude hits its usage limit mid-test**. Three independent layers:
1. **Worker self-stop** (`worker.py`): supervisors (`serve`, and the `run` client) send `.ctl {"op":"ping"}` every 30 s. No ctl for 10 min, or the access token within 5 min of expiry with no fresh one (expiry read from Google's tokeninfo) ⇒ the worker releases its own VM through the Colab API (`colab_api.py`, stdlib).
   - Possible because the CLI's OAuth token (the one already sent to VMs for Drive) also carries the `colaboratory` scope. The VM learns its endpoint from env `COLAB_ENDPOINT` (from the CLI state file, `ColabSession.endpoint`).
   - A warm worker that said `.bye` and was not stopped within 2 min also releases itself.
2. **Supervisors**: idle `.bye` → `colab stop`; heartbeat lost 90 s → stop; SIGTERM → stop all (unchanged).
3. **Reaper** (`guard.py`, `colab-link reap`), from **cron every 5 min on this machine** (installed 2026-10-17; log `~/.cache/colab-link/guard/reaper.log`; compose has a `reaper` service for the server; run it on one machine only, because leases are local).
   - Looks at **every** session on the account (the user chose this, including browser notebooks).
   - Keeps: a worker heartbeat with `supervised_s` < 180, unless idle more than its limit + 10 min or older than 12 h; or an unexpired lease.
   - Everything else is stopped 15 min after it was last seen kept.
   - **Before using Colab by hand (browser, bare `colab new`, probes): `colab-link lease ENDPOINT|--all --minutes N`**, or the reaper stops it.
   - `run --transport exec` renews a 20-min lease on every poll.
- Heartbeats now carry `endpoint`, `supervised_s`, `idle_s`, `idle_limit_s`.
- `idle_minutes` is not part of the pool id: a job with a different `--idle-minutes` lands on an existing warm worker of the same pool.

## LLM serving over NATS (2026-10-17)
- `colab-link llm start|stop|ls MODEL`, `chat MODEL "prompt"`, `openai --port 8000` (local OpenAI-compatible HTTP endpoint).
  - MODEL = a **servable profile** (vLLM venv + `MODEL` in env), e.g. `qwen35-9b-l4`.
- Service endpoints `llm_start`/`llm_stop`/`llm_list`. An LLM pool (`llm-<profile>-<sha8>`, `mode: "llm"`) has no queue: while wanted, the service keeps one worker; that worker runs `vllm serve` on 127.0.0.1:8000 and leaves after `idle_minutes` without requests.
  - LLM workers are never evicted for other pools.
  - Their JIT cache key includes `_mode: llm` (different compiled graphs from the batch job).
- **Wire protocol**: subject `edges.<edge>.colab-link.llm.<model>.{chat.completions,completions,models}`, queue group `llm-<model>`. The payload is the OpenAI request body.
  - Without `stream`: one reply.
  - With `stream: true`: one message per chunk to the reply inbox, then an empty message.
  - Errors carry the header `Nats-Service-Error-Code` (HTTP status) and an OpenAI error body.
  - No worker ⇒ NATS 503 no-responders; clients then call `llm_start` and wait.
- **Text only** (user decision): `--language-model-only` skips the vision encoder. Images/OCR go through batch jobs (`colab-link ocr IMAGES...` builds `tasks.jsonl` for `jobs/vlm_batch.py` and writes one `.txt` per image). NATS max_payload is 1 MB.
- Thinking is off by default (`--default-chat-template-kwargs '{"enable_thinking": false}'`; `ENABLE_THINKING=1` in the profile env turns it on).

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
