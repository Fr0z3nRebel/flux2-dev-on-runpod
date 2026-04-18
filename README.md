# FLUX.2 [dev] on RunPod Serverless

A minimal [RunPod Serverless](https://docs.runpod.io/serverless/overview) worker that accepts a text prompt and generation parameters, runs **FLUX.2 [dev]** through [Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/flux2) (`Flux2Pipeline`), and returns a **PNG as base64** in the job output.

## Prerequisites

1. **Hugging Face**: Log in, open [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev), accept the **FLUX Non-Commercial License**, and create an access token with read access to gated repos.
2. **RunPod**: Account and API key. You can host the image yourself (Docker Hub, GHCR, etc.) **or** use RunPod’s **[GitHub integration](https://docs.runpod.io/serverless/workers/github-integration)** so RunPod builds from your repo and stores the image in their registry—no manual `docker push` required.

Set the token where the worker can read it (RunPod endpoint **Secrets / Environment**):

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`

## Model modes

| `model_mode` | Description | VRAM (rough) |
|--------------|-------------|----------------|
| `bnb4_remote` (default) | `diffusers/FLUX.2-dev-bnb-4bit` + Hugging Face **remote** text encoder | ~18 GB |
| `bnb4_local` | Same 4-bit checkpoint; text encoder runs locally with CPU offload | ~20 GB |
| `full` | `black-forest-labs/FLUX.2-dev` with `enable_model_cpu_offload()` | Often **80 GB** class or heavy offload; see [BFL diffusers doc](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md) |

Override the default with environment variable `MODEL_MODE` on the worker or per-request `model_mode` in the JSON input.

Optional: `REMOTE_TEXT_ENCODER_URL` (default: Hugging Face remote text encoder endpoint used by BFL docs).

## Request `input` fields

| Field | Type | Default | Notes |
|-------|------|---------|--------|
| `prompt` | string | required | |
| `width`, `height` | int | `1024` | Clamped 256–2048 |
| `num_inference_steps` | int | `50` | 1–100 |
| `guidance_scale` | float | `4.0` | 0–20 |
| `seed` | int | omit | Omit for random generation |
| `caption_upsample_temperature` | float | omit | e.g. `0.15` to enable caption upsampling |
| `model_mode` | string | `bnb4_remote` or `MODEL_MODE` env | `full`, `bnb4_remote`, `bnb4_local` |

## Response

Success:

- `image_base64`: PNG bytes, base64-encoded
- `mime_type`: `image/png`
- `generation_time_ms`: integer
- `model_mode`: string
- `seed`: present if you passed `seed`

Errors return `{ "error": "..." }` so the job can finish without raising.

**Payload size**: RunPod `/runsync` responses are limited (see [docs](https://docs.runpod.io/serverless/endpoints/send-requests)); very large PNGs may need a follow-up (e.g. upload to object storage) instead of huge base64 payloads.

## Build and deploy

### Deploy from GitHub (recommended for ongoing work)

RunPod can build and deploy the worker [directly from a GitHub repository](https://docs.runpod.io/serverless/workers/github-integration). **This project** lives at [github.com/Fr0z3nRebel/flux2-dev-on-runpod](https://github.com/Fr0z3nRebel/flux2-dev-on-runpod) (fork it if you want your own copy).

1. In [RunPod Settings → Connections](https://www.console.runpod.io/user/settings), connect **GitHub** and grant access to this repo.
2. **Serverless → New Endpoint → Import Git Repository**, pick the repo, **branch**, and **Dockerfile path** (use `Dockerfile` in the repo root unless you moved it). Use **`rp_handler.py`** at the repo root as the worker entrypoint (same layout as [worker-basic](https://github.com/runpod-workers/worker-basic)).
3. Configure GPU (e.g. **24 GB+** for `bnb4_remote`), timeouts, and add **`HF_TOKEN`** / **`HUGGING_FACE_HUB_TOKEN`** under environment variables.
4. Deploy; monitor the **Builds** tab on the endpoint until the image is **Completed**.

Redeploys are driven by **[GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)** (not every commit). You can also **roll back** to a previous build in the console.

**Limits** (see docs): e.g. build must finish within **160 minutes**, image **≤ 80 GB**, **no private base images** in the Dockerfile, and **no GPU during the build** (install CPU-only wheels in the image; runtime still uses a GPU worker).

### Manual build and registry push

Build for RunPod (`linux/amd64`):

```bash
docker build --platform linux/amd64 -t YOUR_REGISTRY/flux2-dev-runpod:latest .
docker push YOUR_REGISTRY/flux2-dev-runpod:latest
```

In RunPod: **Serverless → New Template** → your image; add **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN`) in the template or endpoint environment. **New Endpoint** → choose a GPU that matches the mode (e.g. **24 GB+** for `bnb4_remote`).

Without **`HF_TOKEN`**, jobs fail with a clear error from the handler (the worker still proves the image and handler run on RunPod).

### Ephemeral image push (no Docker Hub login)

For a quick test you can push to [ttl.sh](https://ttl.sh/) (public, time-limited tag, e.g. `:24h`):

```bash
TAG="ttl.sh/flux2-$(openssl rand -hex 8):24h"
docker tag flux2-dev-runpod:latest "$TAG"
docker push "$TAG"
```

Use `$TAG` as the container image when creating a Serverless template.

### REST API (template + endpoint)

You can create a template and endpoint with the [RunPod REST API](https://docs.runpod.io/api-reference/endpoints/POST/endpoints) (`Authorization: Bearer $RUNPOD_API_KEY`):

- `POST https://rest.runpod.io/v1/templates` — body includes `name`, `imageName`, `isServerless: true`, `category: "NVIDIA"`.
- `POST https://rest.runpod.io/v1/endpoints` — body includes `templateId`, `gpuTypeIds` (e.g. `["NVIDIA GeForce RTX 4090"]`), `executionTimeoutMs` (e.g. `1200000` for cold start + first download), `workersMin`, `workersMax`, `gpuCount`.

Set **`HF_TOKEN`** in the RunPod console on the **template** or **endpoint** so workers can pull gated Hugging Face assets and call the remote text encoder.

## Call the API

Replace `ENDPOINT_ID` and `RUNPOD_API_KEY`. Prefer `/runsync` for a single blocking response (good for interactive use); use `/run` + `/status` for long jobs.

```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A watercolor owl reading a book",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 28,
      "guidance_scale": 4.0,
      "model_mode": "bnb4_remote",
      "seed": 42
    }
  }'
```

## Local testing

With a **CUDA** machine, Hugging Face token, and dependencies installed (or use the Docker image):

1. Copy or keep [`test_input.json`](test_input.json) next to [`rp_handler.py`](rp_handler.py) (same format as production: top-level `"input"`).
2. `export HF_TOKEN=...`
3. Run:

```bash
python rp_handler.py
```

The RunPod Python SDK will pick up `test_input.json` automatically. You can also pass JSON inline:

```bash
python rp_handler.py --test_input '{"input":{"prompt":"Hello","model_mode":"bnb4_remote"}}'
```

For an HTTP test server:

```bash
python rp_handler.py --rp_serve_api
# Then POST to http://localhost:8000/runsync
```

## License and safety

Open **dev** weights are subject to the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md) and Hugging Face gating. The model card discusses **filters / acceptable use**; production systems may need moderation and policy review.
