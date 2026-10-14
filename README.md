# colab-link

Run Python jobs on rented Google Colab GPUs, on demand: start a VM, restore the
environment from a Google Drive cache, run the job, bring back the results, stop
the VM. Built for LLM batch work and OCR with multimodal models (vLLM + Qwen), but
any Python script that follows the [job contract](#job-contract) works.

Two ways to run:

- **One-shot** (`colab-link run`): a fresh VM per job, stopped when the job ends.
- **Service** (`colab-link serve`): a long-running service with a JetStream job
  queue. Workers stay warm between jobs, so a model loads once per VM and later
  jobs take seconds. Idle VMs stop on their own.

```
caller ──① input tar.gz──▶ Google Drive ◀──④ in / out──┐
   │                                                    │
   └─② NATS: run ──▶ colab-link serve ──③ colab CLI: start VM, ship worker──▶ Colab VM (worker.py)
                          ▲                                                    │
                          └──────── NATS: heartbeat / log / progress / result ─┘
```

Files only travel through Google Drive. NATS carries control messages.

## Measured (L4, Qwen3.5-9B fp8, 200 OCR + extraction tasks)

| Case | Submit to result | Model load | CU |
|---|---|---|---|
| Cold VM (caches restored from Drive) | ~6.5 min | 219 s | ~0.17 |
| Warm worker (next job on the same VM) | **33 s** | 0 s | ~0.01 |

## Requirements

- A Colab account with compute units, and the
  [Colab CLI](https://pypi.org/project/google-colab-cli/) logged in (`colab auth`).
  The OAuth token lives in `~/.config/colab-cli/`.
- A GCP project with the **Google Drive API** enabled, used as the quota project
  for Drive calls made with the CLI's own token (`drive.file` scope: colab-link
  only sees files it created). Set `COLAB_LINK_DRIVE_PROJECT=<project>`.
- A NATS server with JetStream, reachable from Colab (for example over `wss://`).
  Following the FlareX edge layout, `~/.flarex/colab-link/` holds:
  - `id`: the edge id; subjects live under `edges.<id>.colab-link.>`
  - `user.creds`: NATS credentials (JWT + nkey seed)
  - Set `NATS_URL` if the server is not `wss://nats.flarex.io`.

The credentials are uploaded to each VM as files (mode 600), never inside executed
code. Use credentials limited to `edges.<id>.colab-link.>`.

## CLI

```bash
bin/colab-link profiles                                  # profiles from profiles.toml
bin/colab-link run qwen35-9b-l4 --input examples/bench   # one-shot
bin/colab-link run --gpu T4 --job my.py --input DIR --pip pandas

bin/colab-link serve                                     # service mode
bin/colab-link submit qwen35-9b-l4 --input DIR --wait    # queue a job, follow it, download output
bin/colab-link submit qwen35-9b-l4 --input DIR           # queue only; prints the job id
bin/colab-link jobs [JOB_ID] | fetch JOB_ID | cancel JOB_ID | workers

bin/colab-link cache ls | cache rm NAME                  # Drive caches
bin/colab-link sessions | stop-all | usage               # Colab sessions and CU balance
```

`bin/colab-link` runs inside the project's uv venv and loads a local `.env` if
there is one.

## Container

```bash
docker compose up -d     # image flarexio/colab-link (linux/amd64, linux/arm64)
```

See `compose.yaml`. It mounts `~/.flarex/colab-link` (read-only), mounts
`~/.config/colab-cli` writable (the CLI refreshes its token there), and keeps VM
state in a volume so that VMs can still be stopped after a restart. On SIGTERM
the service stops every VM it started.

## Service API (NATS micro service `colab-link`)

Subjects live under `edges.<id>.colab-link`, and each request and reply is JSON:

| Endpoint | Request | Reply |
|---|---|---|
| `run` | `{profile? \| gpu?, script?, env?, pip?, cache?, idle_minutes?, in_file}` | job record |
| `status` | `{job_id}` or `{}` | job record, or the most recent jobs |
| `cancel` | `{job_id}` | job record |
| `workers` | `{}` | live workers |

`in_file` is a `.tar.gz` that the caller has already uploaded to the Drive folder
`colab-link-jobs`. When the job finishes, its record carries `out_file`, a
`.tar.gz` in the same folder.

How jobs are scheduled:

- **Pools.** Jobs that share a GPU, script, env, venv and pip set form a pool.
  Each pool has its own JetStream work-queue consumer.
- **Starting VMs.** The service starts a VM for a pool that has jobs waiting. It
  stays within Colab's limits of 2 sessions per GPU type and 5 sessions in total.
- **Stopping VMs.** A VM is stopped when:
  - its worker has been idle for `idle_minutes`,
  - its heartbeat has been missing for 90 seconds, or
  - the service shuts down.
- **Lost workers.** A job running on a lost worker is delivered again.

## Job contract

```
python job.py IN_DIR OUT_DIR
```

- `--input` arrives on the VM as `IN_DIR`.
- Everything the job writes to `OUT_DIR` comes back, together with `log.txt`.
- The job writes `OUT_DIR/progress.json` to report progress.
- Keep the work behind `if __name__ == "__main__":`, because vLLM may spawn
  processes that re-import the main module.

A script can also define `handle(in_dir, out_dir)`. A warm worker then imports the
script once and calls `handle()` for every job, so module globals survive between
jobs. That is how a loaded model is reused. See `jobs/vlm_batch.py`.

## Profiles and caches

`profiles.toml` defines each profile as a GPU, a venv, a job and env. A profile
can also set `idle_minutes` and `max_workers`.

On the first run of a profile, its venv (for example vLLM) and its JIT caches are
built and uploaded to the Drive folder `colab-link-cache`:

- the JIT caches are torch.compile, Triton, FlashInfer and torch inductor;
- cache keys include the package set, the Python and torch versions of the image,
  and the GPU and model settings.

Later VMs restore these caches in about a minute instead of rebuilding them.

## License

MIT
