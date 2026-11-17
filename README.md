# 🛡 scam-lab — a scam radar for the UPI era

**Paste a message → risk score, scam category, red-flag phrases highlighted, and
what to do.** Fine-tuned on the India-specific scam playbook: fake KYC expiry,
"digital arrest", electricity bill cut-offs, lottery/KBC calls, loan-app
blackmail, courier-with-customs, income-tax refund traps — in English *and*
Hinglish, which generic spam filters routinely miss.

## Why this is unique (the honest pitch)

Spam classifiers are a dime a dozen. This is different in three ways:

1. **Domain-specific taxonomy** — 10 scam families with their actual scripts,
   seed corpus hand-curated (see `src/seeds.jsonl`), not generic "spam vs ham".
2. **Hinglish-aware** — scam messages code-switch mid-sentence; I trained on
   that reality.
3. **Calibrated, not just accurate** — the worst failure isn't missing a scam,
   it's flagging your *real* bank OTP as fraud. The eval reports the
   false-positive rate on benign messages separately, plus calibration bins.
   (Same rigor as my judge-lab: metrics you can act on.)

## The practical application

| User | Use |
|---|---|
| Anyone | Paste a suspicious SMS/WhatsApp → verdict + what to do, in 2 seconds |
| Families | Share as a "before you tap that link" checker for parents |
| Fintech/banks | This exact pipeline, fed by live message streams, is a fraud-ops product |
| Analysts | The red-flag taxonomy is machine-readable — aggregrable across messages |

## What's inside

```
src/seeds.jsonl        ← hand-curated seed corpus: scam taxonomy + benign probes
src/build_dataset.py   ← stdlib-only: merge seeds + external corpora, dedup, split
src/evaluate.py        ← stdlib-only: AUC, scam-type F1, benign-FPR, calibration
notebooks/train_….ipynb← Colab free T4: augment seeds, LoRA-train Qwen3-0.6B, push to HF
space/app.py           ← HF Space: paste a message → verdict with highlighted red flags
frontend/              ← Cloudflare Pages UI → talks to the Space
```

## Your 90-minute path (all free)

1. **Colab** (`notebooks/train_scam_detector.ipynb`): T4 GPU, Run all. It
   augments the hand-curated seeds with paraphrases, trains a LoRA on
   Qwen3-0.6B to output structured verdicts, and evaluates against the
   held-out split.
2. **Publish** the model to HF with the eval table in the model card.
3. **Space**: upload `space/`, set `MODEL_REPO` → live demo.
4. **Frontend**: deploy `frontend/` to Cloudflare Pages, point it at the Space.

## Cost: $0 — Colab free T4, HF Hub, HF Spaces CPU, Cloudflare Pages.

## Interview talking points

- **Why generative JSON verdicts instead of a classifier?** The verdict carries
  the taxonomy: category, red-flag spans, action. One model, structured output,
  and the explanation is auditable — you can check *why* it flagged.
- **What's the metric that matters?** Benign-FPR. A scam detector that flags
  your real OTP message trains users to ignore it — worse than useless.
- **How do you handle drift?** Scammers rephrase weekly; the seed taxonomy +
  paraphrase augmentation makes retraining a 30-minute Colab job, not a
  data-labelling project.
- **Known failure mode:** brand-new scam scripts the taxonomy hasn't seen —
  measured as scam-type F1 falling on out-of-taxonomy holds in the report.

## Honest limits

- Seed corpus is hand-curated (~70 messages) + augmentation — real-world scale
  needs live traffic; the pipeline is built for that.
- English + Hinglish only; other Indic scripts are future work.
- Not affiliated with any bank; treat output as advice, not a guarantee.
