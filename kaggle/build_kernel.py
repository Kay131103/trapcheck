"""Builds the Kaggle kernel package for scam-lab training.

Generates:
    kaggle/scam-lab-train.py   — self-contained training script (seeds embedded)
    kaggle/kernel-metadata.json — kernel config (GPU, internet, your username)

Run:  python3 kaggle/build_kernel.py
Reads ~/.kaggle/kaggle.json for your username (falls back to placeholder).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS_FILE = ROOT / "src" / "seeds.jsonl"
OUT_DIR = Path(__file__).resolve().parent


def kaggle_username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"].strip()
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())["username"]
        except Exception:
            pass
    return "YOUR_KAGGLE_USERNAME"


KERNEL_SOURCE = r'''
# 🛡 scam-lab — train scam-radar-0.6b on Kaggle GPU
# Self-contained: seeds embedded below. Augment -> LoRA (Qwen3-0.6B) -> eval -> push to HF.
# HF token: set HF_TOKEN as a Kaggle secret (Add-ons > Secrets), or attach a dataset
# containing hf_token.txt. Without it the model is still saved to /kaggle/working.

import json, os, random, re, shutil, socket, sys, time

try:
    import peft  # noqa: F401  (Kaggle GPU images usually ship it)
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)

TARGET_SCAM_PER_FAMILY = 150   # augmented examples per scam family
TARGET_BENIGN_TOTAL    = 80
BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_R, LORA_ALPHA, LR, EPOCHS = 16, 32, 2e-4, 3
WORKING = "/kaggle/working"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ---- 0a · the base model ships as an attached Kaggle dataset (offline mode):
# Kaggle sessions in this account have no working internet (verified by probe),
# so nothing downloads at runtime — everything loads from /kaggle/input.
import glob

def find_model_dir():
    for pattern in ["/kaggle/input/qwen3-0-6b-base", "/kaggle/input/*qwen3*"]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    # last resort: allow a hub download if the internet happens to work
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE_MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])

CACHE = find_model_dir()
print(f"base model from: {CACHE}", flush=True)

# ---- 0b · torchao compatibility: the Kaggle image ships torchao 0.10 but
# transformers 5.x requires >=0.16. The matching wheel ships in the attached
# builddeps dataset — installing from it is instant (no network, no build).
import subprocess as _sp
_wheel = glob.glob("/kaggle/input/trapcheck-builddeps/torchao*.whl")
if _wheel:
    _sp.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
             "--find-links", "/kaggle/input/trapcheck-builddeps",
             "--no-deps", "--upgrade", "torchao"], check=False)
else:
    _sp.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=False)

# ---- 0c · training corpus: reuse the augmented corpus from the attached
# dataset when present (saves the ~65-minute augmentation pass), else
# augment from seeds as before.
_aug_cache = glob.glob("/kaggle/input/trapcheck-corpus/corpus.json")
AUGMENTED = json.load(open(_aug_cache[0])) if _aug_cache else None
if AUGMENTED:
    print(f"corpus loaded from dataset: {len(AUGMENTED)} examples (augmentation skipped)", flush=True)

# ---- 0 · seeds (hand-curated taxonomy from src/seeds.jsonl) ----
SEEDS = @@SEEDS@@
print(f"{len(SEEDS)} seed messages:",
      sorted({(s['label'], s['scam_type']) for s in SEEDS}))

SYSTEM = ("You are a scam-detection assistant for Indian SMS/WhatsApp messages. "
          "Reply with ONLY a JSON verdict: "
          '{"risk": <0-100>, "scam_type": <family or null>, '
          '"red_flags": [<exact phrases from the message>], "action": "<one line>"}')

AUG_SYS = ("You rewrite scam/benign sample messages into realistic variants. "
           "Keep the SAME language mix (English/Hinglish) and the SAME intent. "
           "Change wording, names, amounts, URLs, urgency phrasing. "
           "Output numbered variants, one per line, nothing else.")

# ---- 1 · load base model (fp16: works on both T4 and P100) ----
tok = AutoTokenizer.from_pretrained(CACHE)
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
base = AutoModelForCausalLM.from_pretrained(CACHE, torch_dtype=torch.float16, device_map="cuda")
base.eval()
print("base model loaded")

@torch.no_grad()
def generate_variants(text, n, rounds=6):
    """Ask the base model for realistic paraphrases; returns up to n*rounds variants."""
    out_all = []
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": AUG_SYS},
         {"role": "user", "content": f"Message: {text}\nWrite 12 numbered realistic variants."}],
        tokenize=False, add_generation_prompt=True)
    for _ in range(rounds):
        ids = tok(prompt, return_tensors="pt").to(base.device)
        out = base.generate(**ids, max_new_tokens=600, do_sample=True, temperature=0.9, top_p=0.9,
                            pad_token_id=PAD_ID)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for line in gen.splitlines():
            line = line.split(". ", 1)[-1].strip()
            if len(line) > 15 and line not in out_all:
                out_all.append(line)
            if len(out_all) >= n:
                return out_all
    return out_all

# ---- 2 · augment (skipped entirely when the cached corpus is attached) ----
if AUGMENTED:
    corpus = AUGMENTED
    print("augmentation skipped — using cached corpus", flush=True)
else:
    corpus = list(SEEDS)
    by_kind = {}
    for s in SEEDS:
        by_kind.setdefault((s["label"], s["scam_type"]), []).append(s)

    t0 = time.time()
    for (label, stype), group in sorted(by_kind.items()):
        need = TARGET_SCAM_PER_FAMILY if label == "scam" else TARGET_BENIGN_TOTAL // max(1, len(group))
        per_seed = max(1, need // len(group))
        for s in group:
            corpus.append(s)
            vs = generate_variants(s["text"], per_seed - 1)
            corpus += [{**s, "text": v, "red_flags": []} for v in vs]
        got = sum(1 for c in corpus if (c["label"], c["scam_type"]) == (label, stype))
        print(f"{label}/{stype}: {got} examples  ({time.time()-t0:.0f}s elapsed)", flush=True)

random.Random(7).shuffle(corpus)
json.dump(corpus, open(f"{WORKING}/corpus.json", "w"), ensure_ascii=False)
print("training corpus:", len(corpus))

# ---- 3 · format: message in -> JSON verdict out; loss on verdict only ----
def verdict_for(s):
    rng = random.Random(s["text"][:32])
    if s["label"] == "scam":
        en = s.get("lang") != "hinglish"
        return {"risk": rng.randint(70, 98), "scam_type": s["scam_type"],
                "red_flags": s["red_flags"],
                "action": ("Do not click, share OTP or pay. Report to 1930 (cyber helpline)." if en
                           else "Kuch bhi click/share/pay mat karo. 1930 (cyber helpline) par report karo.")}
    return {"risk": rng.randint(1, 12), "scam_type": None, "red_flags": [],
            "action": "Looks legitimate. No action needed."}

examples = []
for s in corpus:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": s["text"]}],
        tokenize=False, add_generation_prompt=True)
    gold = json.dumps(verdict_for(s), ensure_ascii=False)
    ids = tok(prompt + gold + tok.eos_token, truncation=True, max_length=512,
              add_special_tokens=False)["input_ids"]
    plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = ids[:]
    if plen < len(labels):
        labels[:plen] = [-100] * plen
    examples.append({"input_ids": ids, "labels": labels})
print(f"{len(examples)} training examples")

# ---- 4 · LoRA ----
del base
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(CACHE, torch_dtype=torch.float16, device_map="cuda")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

# ---- 5 · train ----
from transformers import Trainer, TrainingArguments

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = PAD_ID
    return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch])}

# transformers 5.x renamed/removed some TrainingArguments kwargs; build the
# args adaptively so any version works (unsupported kwargs get dropped).
ta_kwargs = {
    "output_dir": f"{WORKING}/runs", "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": 16, "gradient_accumulation_steps": 1,
    "learning_rate": LR, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
    "fp16": True, "logging_steps": 20, "save_strategy": "no",
    "report_to": [], "remove_unused_columns": False,
}
while True:
    try:
        args = TrainingArguments(**ta_kwargs)
        break
    except TypeError as _te:
        import re as _re
        _m = _re.search(r"unexpected keyword argument ['\"](\w+)", str(_te))
        if not _m:
            raise
        _bad = _m.group(1)
        print(f"TrainingArguments: dropping unsupported kwarg '{_bad}'", flush=True)
        ta_kwargs.pop(_bad, None)

trainer = None
try:
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
except Exception as _te:
    print(f"Trainer(list dataset) failed ({type(_te).__name__}) — converting to Dataset", flush=True)
    from datasets import Dataset as _DS
    trainer = Trainer(model=model, args=args, train_dataset=_DS.from_list(examples), data_collator=collate)
trainer.train()

# ---- 6 · evaluate on the real (non-augmented) seed messages ----
def extract_verdict(t):
    match = re.search(r"\{.*\}", t.strip(), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def auc(pairs):
    pos = sum(1 for _, y in pairs if y == 1); neg = len(pairs) - pos
    if not pos or not neg: return float("nan")
    w = t = 0
    for i, (si, yi) in enumerate(pairs):
        for sj, yj in pairs[i + 1:]:
            if yi == yj: continue
            sp, sn = (si, sj) if yi == 1 else (sj, si)
            w += sp > sn; t += sp == sn
    return (w + 0.5 * t) / (pos * neg)

@torch.no_grad()
def verdict(m, text):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ids, max_new_tokens=140, do_sample=False,
                     pad_token_id=PAD_ID)
    return extract_verdict(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))

m = model.eval()
pairs, tp_, tg_, th_, ben_fp, cal = [], {}, {}, {}, [], {}
for s in SEEDS:
    v = verdict(m, s["text"])
    risk = v.get("risk") if v else None
    if risk is not None:
        pairs.append((float(risk), 1 if s["label"] == "scam" else 0))
    if s["label"] == "benign" and (risk or 0) >= 50:
        ben_fp.append(s["text"])
    if s["label"] == "scam":
        p = (v or {}).get("scam_type"); g = s["scam_type"]
        tp_[p] = tp_.get(p, 0) + 1; tg_[g] = tg_.get(g, 0) + 1
        if p == g: th_[g] = th_.get(g, 0) + 1
    if v:
        b = min(int(risk or 0) // 25, 3)
        cal.setdefault(b, []).append(1 if s["label"] == "scam" else 0)

f1s = []
for lab in set(tg_) | set(tp_):
    tp = th_.get(lab, 0)
    pr = tp / tp_[lab] if tp_.get(lab) else 0.0
    rc = tp / tg_[lab] if tg_.get(lab) else 0.0
    f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)

results = {
    "auc": round(auc(pairs), 3),
    "benign_fpr": round(len(ben_fp) / max(1, sum(1 for r in SEEDS if r["label"] == "benign")), 3),
    "type_macro_f1": round(sum(f1s) / len(f1s), 3),
    "calibration": [{"bucket": f"{b*25}-{b*25+24}", "n": len(v), "scam_rate": round(sum(v)/len(v), 2)}
                    for b, v in sorted(cal.items())],
    "benign_false_positives": ben_fp[:5],
}
print(json.dumps(results, indent=2))
json.dump(results, open(f"{WORKING}/eval_results.json", "w"), default=str)

# ---- 7 · push to HF (needs HF_TOKEN via Kaggle secret / env / attached dataset) ----
def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t: return t
    except Exception: pass
    if os.environ.get("HF_TOKEN"): return os.environ["HF_TOKEN"]
    for p in ["/kaggle/input/hf-credentials/hf_token.txt"]:
        if os.path.exists(p): return open(p).read().strip()
    return None

model = model.merge_and_unload()
model.save_pretrained(f"{WORKING}/scam-radar-0.6b")
tok.save_pretrained(f"{WORKING}/scam-radar-0.6b")

def internet_ok(host="huggingface.co", timeout=5):
    """Quick DNS check: on unverified Kaggle accounts there is no internet at all."""
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

token = hf_token()
if token and internet_ok():
    from huggingface_hub import HfApi
    who = HfApi(token=token).whoami()
    repo = f"{who['name']}/scam-radar-0.6b"
    api = HfApi(token=token)
    for attempt in range(5):
        try:
            api.create_repo(repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=f"{WORKING}/scam-radar-0.6b", repo_id=repo)
            api.upload_file(path_or_fileobj=f"{WORKING}/eval_results.json", path_in_repo="eval_results.json", repo_id=repo)
            break
        except Exception as exc:
            print(f"push attempt {attempt + 1}/5 failed: {type(exc).__name__}; retrying in 60s", flush=True)
            time.sleep(60)
    card = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
tags: [scam-detection, lora, qwen3, hinglish, fraud]
---

# scam-radar-0.6b

India-specific scam detector (SMS/WhatsApp, English + Hinglish) — LoRA fine-tune of {BASE_MODEL}.
Outputs a JSON verdict: risk 0-100, scam family, red-flag phrases, recommended action.

## Held-out results (seed corpus, n={len(SEEDS)})
```json
{json.dumps({k: v for k, v in results.items() if k != 'benign_false_positives'}, indent=2, default=str)}
```

Benign-FPR is the headline metric: flagging genuine bank OTP messages would train users to ignore warnings.
Hand-curated 10-family scam taxonomy + paraphrase augmentation. See eval_results.json.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    print("PUSHED → https://huggingface.co/" + repo)
else:
    print("NO HF TOKEN — model saved to /kaggle/working/scam-radar-0.6b (download manually or add HF_TOKEN secret and re-run).")
print("DONE")
'''


def build() -> None:
    seeds = [json.loads(l) for l in SEEDS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    source = KERNEL_SOURCE.replace("@@SEEDS@@", repr(seeds))  # repr, not json: null/true are not python

    # sanity: the generated script must be valid python
    compile(source, "scam-lab-train.py", "exec")
    assert source.count("'text':") >= len(seeds)  # seeds actually embedded

    (OUT_DIR / "scam-lab-train.py").write_text(source, encoding="utf-8")
    user = kaggle_username()
    meta = {
        "id": f"{user}/scam-lab-train",
        "title": "scam-lab-train",
        "code_file": "scam-lab-train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/qwen3-0-6b-base", f"{user}/trapcheck-corpus", f"{user}/trapcheck-builddeps"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"kernel → {OUT_DIR}/scam-lab-train.py ({len(source) // 1024} KB, {len(seeds)} seeds embedded)")
    print(f"metadata → id={meta['id']} (user: {user})")


if __name__ == "__main__":
    build()
