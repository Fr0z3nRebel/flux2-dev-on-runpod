"""
RunPod Serverless worker: FLUX.2 [dev] text-to-image via Diffusers.

Model modes (env MODEL_MODE or input.model_mode):
  full        — black-forest-labs/FLUX.2-dev + enable_model_cpu_offload (large VRAM or offload)
  bnb4_remote — diffusers/FLUX.2-dev-bnb-4bit + Hugging Face remote text encoder (~18G VRAM)
  bnb4_local  — same repo, local 4-bit text encoder + CPU offload (~20G VRAM)

Requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) with access to gated FLUX.2-dev weights.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Any

import requests
import runpod
import torch
from diffusers import AutoModel, Flux2Pipeline
from huggingface_hub import get_token
from PIL import Image
from transformers import Mistral3ForConditionalGeneration

# --- Config ---

DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
DEFAULT_MODEL_MODE = os.environ.get("MODEL_MODE", "bnb4_remote")
REMOTE_ENCODER_URL = os.environ.get(
    "REMOTE_TEXT_ENCODER_URL",
    "https://remote-text-encoder-flux-2.huggingface.co/predict",
)

REPO_FULL = "black-forest-labs/FLUX.2-dev"
REPO_BNB4 = "diffusers/FLUX.2-dev-bnb-4bit"

MIN_SIZE = 256
MAX_SIZE = 2048
DEFAULT_SIZE = 1024
MIN_STEPS = 1
MAX_STEPS = 100
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0

VALID_MODES = frozenset({"full", "bnb4_remote", "bnb4_local"})

_pipe: Flux2Pipeline | None = None
_pipe_mode: str | None = None
_pipe_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def remote_text_encoder(prompt: str, device: str) -> torch.Tensor:
    token = _get_hf_token() or get_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
            "(and accept the FLUX.2-dev license on Hugging Face)."
        )
    response = requests.post(
        REMOTE_ENCODER_URL,
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    response.raise_for_status()
    buf = io.BytesIO(response.content)
    # Embeddings from HF endpoint; not weights-only tensor format
    prompt_embeds = torch.load(buf, map_location="cpu", weights_only=False)
    return prompt_embeds.to(device)


def _load_pipeline(model_mode: str) -> Flux2Pipeline:
    torch_dtype = torch.bfloat16

    if model_mode == "full":
        pipe = Flux2Pipeline.from_pretrained(
            REPO_FULL,
            torch_dtype=torch_dtype,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    if model_mode == "bnb4_remote":
        pipe = Flux2Pipeline.from_pretrained(
            REPO_BNB4,
            text_encoder=None,
            torch_dtype=torch_dtype,
        )
        pipe.to(DEVICE)
        return pipe

    if model_mode == "bnb4_local":
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            REPO_BNB4,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        dit = AutoModel.from_pretrained(
            REPO_BNB4,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        pipe = Flux2Pipeline.from_pretrained(
            REPO_BNB4,
            text_encoder=text_encoder,
            transformer=dit,
            torch_dtype=torch_dtype,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    raise ValueError(f"Unknown model_mode: {model_mode}")


def get_pipeline(model_mode: str) -> Flux2Pipeline:
    global _pipe, _pipe_mode
    with _pipe_lock:
        if _pipe is not None and _pipe_mode == model_mode:
            return _pipe
        _pipe = _load_pipeline(model_mode)
        _pipe_mode = model_mode
        return _pipe


def validate_input(data: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return None, "Missing or invalid 'prompt' (non-empty string required)."
    if len(prompt) > 10000:
        return None, "'prompt' exceeds maximum length."

    width = int(data.get("width", DEFAULT_SIZE))
    height = int(data.get("height", DEFAULT_SIZE))
    if not (MIN_SIZE <= width <= MAX_SIZE and MIN_SIZE <= height <= MAX_SIZE):
        return None, f"width/height must be between {MIN_SIZE} and {MAX_SIZE}."

    steps = int(data.get("num_inference_steps", DEFAULT_STEPS))
    if not (MIN_STEPS <= steps <= MAX_STEPS):
        return None, f"num_inference_steps must be between {MIN_STEPS} and {MAX_STEPS}."

    guidance = float(data.get("guidance_scale", DEFAULT_GUIDANCE))
    if not (0.0 <= guidance <= 20.0):
        return None, "guidance_scale must be between 0 and 20."

    model_mode = data.get("model_mode", DEFAULT_MODEL_MODE)
    if model_mode is None:
        model_mode = DEFAULT_MODEL_MODE
    if not isinstance(model_mode, str):
        return None, "model_mode must be a string."
    model_mode = model_mode.strip().lower()
    if model_mode not in VALID_MODES:
        return None, f"model_mode must be one of: {', '.join(sorted(VALID_MODES))}."

    cap_temp = data.get("caption_upsample_temperature")
    if cap_temp is not None:
        cap_temp = float(cap_temp)

    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)

    return {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "model_mode": model_mode,
        "caption_upsample_temperature": cap_temp,
        "seed": seed,
    }, None


def image_to_png_base64(image: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/png"


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        return {"error": "Invalid job: 'input' must be an object."}

    params, err = validate_input(job_input)
    if err:
        return {"error": err}

    assert params is not None
    model_mode = params["model_mode"]
    t0 = time.perf_counter()

    try:
        pipe = get_pipeline(model_mode)
    except Exception as e:
        return {"error": f"Failed to load pipeline: {e!s}"}

    call_kw: dict[str, Any] = {
        "width": params["width"],
        "height": params["height"],
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params["guidance_scale"],
    }
    if params["seed"] is not None:
        call_kw["generator"] = torch.Generator(device=DEVICE).manual_seed(params["seed"])
    if params["caption_upsample_temperature"] is not None:
        call_kw["caption_upsample_temperature"] = params["caption_upsample_temperature"]

    try:
        if model_mode == "bnb4_remote":
            call_kw["prompt_embeds"] = remote_text_encoder(params["prompt"], DEVICE)
        else:
            call_kw["prompt"] = params["prompt"]

        with _infer_lock:
            out = pipe(**call_kw)
        image = out.images[0]
    except Exception as e:
        return {"error": f"Generation failed: {e!s}"}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    b64, mime = image_to_png_base64(image)

    result: dict[str, Any] = {
        "image_base64": b64,
        "mime_type": mime,
        "generation_time_ms": elapsed_ms,
        "model_mode": model_mode,
    }
    if params["seed"] is not None:
        result["seed"] = params["seed"]
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
