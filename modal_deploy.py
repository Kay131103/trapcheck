"""Modal deployment for TrapCheck 0.6B.

Deploy:  modal deploy modal_deploy.py
Result:  a public URL serving POST /verdict {message} → JSON verdict,
         running the merged Qwen3-0.6B scam detector on Modal's free credits.
"""

import modal

MODEL_DIR = "/model"
SYSTEM = (
    "You are a scam-detection assistant for Indian SMS/WhatsApp messages. "
    "Reply with ONLY a JSON verdict: "
    '{"risk": <0-100>, "scam_type": <family or null>, '
    '"red_flags": [<exact phrases from the message>], "action": "<one line>"}'
)


def _download_model():
    """Build step: bake the model into the image so cold starts skip HF."""
    from huggingface_hub import snapshot_download

    snapshot_download("kitty1901/trapcheck-0.6b", local_dir=MODEL_DIR)
    print("model baked into image")


# Match the training environment (Kaggle: transformers 5.20.2, torch 2.11) —
# the saved tokenizer config uses the 5.x format that 4.x cannot parse.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==5.20.2",
        "accelerate>=1.15",
        "huggingface_hub>=0.30",
        "fastapi>=0.110",
        "uvicorn>=0.29",
    )
    .run_function(_download_model, timeout=1800)
)

app = modal.App("trapcheck")


@app.function(image=image, scaledown_window=600, cpu=2, memory=6144)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    """Runs on container start: loads the model, returns the ASGI app."""
    import json
    import re

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os

    print("loading model…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.eval()
    print("model ready", flush=True)

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
    )

    JSON_RE = re.compile(r"\{[\s\S]*\}")

    def extract_verdict(text):
        m = JSON_RE.search((text or "").strip())
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok", "model": "trapcheck-0.6b"}

    @fastapi_app.post("/verdict")
    async def verdict(req: Request):
        body = await req.json()
        message = (body.get("message") or "").strip()[:1000]
        if not message:
            return {"error": "empty message"}

        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt")
        out = model.generate(
            **ids, max_new_tokens=180, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        v = extract_verdict(raw)
        if not v:
            return {"error": "unparseable verdict", "raw": raw[:200]}
        return v

    return fastapi_app
