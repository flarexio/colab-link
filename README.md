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
docker compose up -d     # service + HTTP API + reaper; image flarexio/colab-link (amd64, arm64)
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

## HTTP API

`colab-link api` puts an HTTP front on the service: it takes uploads, stores them
in Drive, queues the job, and serves the outputs back. It also answers the
OpenAI routes under `/v1` (see below).

```bash
bin/colab-link api --port 8080 --api-key "$KEY"      # binding off loopback needs a key
H="Authorization: Bearer $KEY"

# a job: profile (or gpu + script), env KEY=VALUE, pip, idle_minutes;
# files (repeatable; the filename may include sub/dirs) and/or archive (.tar.* or .zip)
curl -H "$H" -F profile=qwen35-9b-l4 -F files=@tasks.jsonl -F archive=@imgs.tar.gz \
     http://127.0.0.1:8080/api/jobs                        # -> 202 + job record with links
curl -H "$H" -N http://127.0.0.1:8080/api/jobs/$ID/events  # SSE: status, log, progress, end
curl -H "$H" "http://127.0.0.1:8080/api/jobs/$ID?wait=600" # or long-poll until it ends
curl -H "$H" -o out.tar.gz http://127.0.0.1:8080/api/jobs/$ID/output

# OCR: images in, one result per image out
curl -H "$H" -F images=@scan1.png -F images=@scan2.jpg http://127.0.0.1:8080/api/ocr
curl -H "$H" http://127.0.0.1:8080/api/jobs/$ID/results   # [{"id": "scan1.png", "output": "..."}, ...]
```

| Route | |
|---|---|
| `GET /api/health` | no key needed |
| `GET /api/profiles` | profiles and whether they can be served as LLMs |
| `POST /api/jobs` | multipart (above), or JSON `{profile, ...}` for a job with no input |
| `GET /api/jobs` · `GET /api/jobs/{id}[?wait=s]` · `DELETE /api/jobs/{id}` | list · status · cancel |
| `GET /api/jobs/{id}/events` | server-sent events until the job ends |
| `GET /api/jobs/{id}/output` · `/files` · `/files/{path}` · `/results` | outputs |
| `POST /api/ocr` | `images` (repeatable), `profile`, `prompt`, `max_tokens`, `idle_minutes` |
| `GET /api/workers` · `GET /api/usage` | warm workers · CU balance |
| `GET /api/llm` · `POST /api/llm/{model}/start[?wait=1]` · `POST /api/llm/{model}/stop` | model servers |

Errors are JSON: `{"error": {"message", "code"}}`. The first download of a job's
output fetches it from Drive, deletes the job's Drive files, and keeps a local
copy for two days (`COLAB_LINK_API_DATA`). Uploads are streamed to disk, up to
`--max-upload-mb` (2048). Put a TLS reverse proxy in front of it when it is
reachable from outside.

## LLMs over NATS (OpenAI-compatible)

A servable profile (a vLLM venv and a `MODEL` in its env) can also run as a
long-lived model server. The worker runs `vllm serve` on the VM and relays
OpenAI requests between NATS and the model:

```bash
bin/colab-link llm start qwen35-9b-l4 --wait     # cold start takes a few minutes
bin/colab-link chat qwen35-9b-l4 "Summarise ..."  # streams; starts the model if needed
bin/colab-link openai --port 8000                 # local OpenAI HTTP endpoint (or use `api`)
bin/colab-link llm ls | llm stop qwen35-9b-l4
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen35-9b-l4", messages=[{"role": "user", "content": "hi"}])
```

Wire protocol, for clients that speak NATS directly:

- **Subject.** `edges.<id>.colab-link.llm.<model>.{chat.completions,completions,models}`.
  The payload is the OpenAI request body. Workers share a queue group, so
  several VMs split the load.
- **Replies.** A normal request gets one reply. With `"stream": true`, each
  chunk is a separate message to the reply subject, and an empty message ends
  the stream.
- **Errors.** An error reply carries the header `Nats-Service-Error-Code` (the
  HTTP status) and an OpenAI error body.
- **No worker.** If no worker serves the model, NATS answers 503 (no
  responders). The client then calls the service endpoint `llm_start
  {model}` and retries.

Chat is text only. Images go through batch jobs:
`bin/colab-link ocr scans/ --out results/ocr` sends the images as one batch job
and writes one `.txt` per image.

The model server stops after the profile's `idle_minutes` without requests.

## Idle VMs

Colab bills a VM while it is assigned, so colab-link has three independent ways
to stop VMs nobody is using:

1. **The VM stops itself.** Supervisors ping each worker every 30 seconds. If a
   worker hears nothing for 10 minutes, or its access token is about to expire,
   it releases its own VM through the Colab API. This works even if the machine
   running colab-link is gone.
2. **The service stops VMs** whose worker is idle or whose heartbeat is
   missing, and it stops all of its VMs when it shuts down.
3. **The reaper** (`colab-link reap`, from cron, or the `reaper` service in
   `compose.yaml`) checks every session on the account:
   - it keeps sessions whose worker is supervised, and sessions with a lease;
   - it stops any other session after 15 minutes;
   - even a supervised session is stopped once it has been idle 10 minutes past
     its idle limit, or has run 12 hours without a lease.

   Before you use Colab by hand (for example in a browser), take a lease with
   `colab-link lease --all --minutes 120`.

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
