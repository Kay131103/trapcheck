"""scam-lab demo: paste an SMS/WhatsApp message → structured verdict with
highlighted red-flag phrases. Runs the fine-tuned 0.6B model on free CPU."""

from __future__ import annotations

import json
import os
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = os.environ.get("MODEL_REPO", "kitty1901/scam-radar-0.6b")

SYSTEM = (
    "You are a scam-detection assistant for Indian SMS/WhatsApp messages. "
    "Reply with ONLY a JSON verdict: "
    '{"risk": <0-100>, "scam_type": <family or null>, '
    '"red_flags": [<exact phrases from the message>], "action": "<one line>"}'
)

FAMILY_LABELS = {
    "kyc_expiry": "Fake KYC expiry",
    "digital_arrest": "Digital arrest / fake police",
    "utility_bill": "Electricity bill disconnection",
    "lottery": "Lottery / prize / government grant",
    "loan_app": "Loan app / instant credit",
    "job_scam": "Job / work-from-home / exam paper",
    "courier_customs": "Courier stuck at customs",
    "otp_fraud": "OTP / CVV harvesting",
    "investment_tips": "Trading / crypto tips",
    "tax_refund": "Tax / subsidy refund",
    "emergency_scam": "Emergency / sextortion",
}


def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, torch_dtype=torch.float32)
    model.eval()
    return tok, model


print(f"loading {MODEL_REPO} …")
tok, model = _load()
print("ready")


def extract_verdict(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def highlight(text: str, flags: list[str]) -> str:
    """Escape the message but wrap exact red-flag spans in <mark> — works even
    when a flag contains characters that need escaping (&, <, >)."""
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts: list = [text]  # plain text chunks and ("FLAG", phrase) tuples
    for flag in flags or []:
        if not flag:
            continue
        rebuilt = []
        for part in parts:
            if isinstance(part, tuple):
                rebuilt.append(part)
                continue
            chunks = part.split(flag)
            for i, chunk in enumerate(chunks):
                if i:
                    rebuilt.append(("FLAG", flag))
                if chunk:
                    rebuilt.append(chunk)
        parts = rebuilt
    out = []
    for part in parts:
        if isinstance(part, tuple):
            out.append(
                f'<mark style="background:#7f1d1d;color:#fecaca;padding:0 3px;border-radius:3px">{esc(part[1])}</mark>'
            )
        else:
            out.append(esc(part))
    return "".join(out)


def analyze(message: str):
    if not message.strip():
        return "Paste a message above.", ""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=160, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    v = extract_verdict(raw)
    if not v:
        return f"(unparseable verdict) {raw[:160]}", ""

    risk = v.get("risk", 0)
    if not isinstance(risk, (int, float)):
        risk = 0
    level = "🔴 HIGH" if risk >= 70 else "🟠 MEDIUM" if risk >= 40 else "🟢 LOW"
    family = FAMILY_LABELS.get(v.get("scam_type") or "", "—")
    verdict = f"### {level} — risk {int(risk)}/100\n**Category:** {family}\n\n**Action:** {v.get('action', '—')}"
    trace = (
        "### 🔍 red flags found in the message\n"
        + ("\n".join(f"- `{f}`" for f in v.get("red_flags", [])) or "_none extracted_")
        + f"\n\n### message with flags\n> {highlight(message.strip(), v.get('red_flags', []))}"
        + f"\n\n### model output\n```json\n{json.dumps(v, indent=1, ensure_ascii=False)}\n```"
    )
    return verdict, trace


with gr.Blocks(title="scam-lab") as demo:
    gr.Markdown(
        "## 🛡 scam-lab — paste a suspicious message\n"
        "India/UPI-specific scam radar (English + Hinglish). Risk score, scam family, "
        "and the exact red-flag phrases highlighted. A benign-FPR-first detector: it is "
        "trained not to cry wolf on genuine bank messages.\n\n"
        "Try the examples — the last two are *real* messages people get every day."
    )
    msg = gr.Textbox(lines=4, label="Message", placeholder="Paste the SMS / WhatsApp text here…")
    with gr.Row():
        go = gr.Button("Analyze", variant="primary")
        clear = gr.Button("Clear")
    verdict = gr.Markdown()
    trace = gr.Markdown()

    examples = gr.Examples(
        examples=[
            ["Dear customer your SBI KYC will expire today. Click sbi-kyc-update[.]site to update else account blocked permanently"],
            ["This is Mumbai cyber police. Your number is linked to money laundering case. Digital arrest order issued. Pay Rs 50,000 security deposit immediately via UPI"],
            ["Aapka Paytm KYC aaj raat 12 baje tak update nahi hua toh wallet block ho jayega. Link: paytm-kyc[.]xyz"],
            ["Dear customer, OTP 483921 for Rs 1,250 spent on Flipkart via HDFC card xx8821. NOT share with anyone -HDFC Bank"],
            ["Your electricity bill of Rs 1,140 for Oct 2026 is generated. Due date 20-11-2026. Pay on official app or website."],
        ],
        inputs=[msg],
    )

    go.click(analyze, [msg], [verdict, trace])
    msg.submit(analyze, [msg], [verdict, trace])
    clear.click(lambda: ("", "", ""), outputs=[msg, verdict, trace])

demo.launch()
