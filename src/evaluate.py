"""Score a scam-detector model on the held-out test set. stdlib + huggingface_hub.

The model under test receives a message and must answer with a JSON verdict:
    {"risk": 0-100, "scam_type": <family|null>, "red_flags": [...], "action": "..."}

Metrics (and why they matter):
  auc          — ranking quality using `risk` as the score
  benign_fpr   — share of BENIGN messages flagged risk>=50. THE key metric:
                 flagging a real bank OTP trains users to ignore warnings
  type_f1      — scam family accuracy on scam messages
  calibration  — predicted mean risk vs. actual scam rate per bucket; honest
                 detectors say 20% risk when 20% of such messages are scams

Usage:
  python -m src.evaluate --model <hf-model-id> --test data/test.jsonl [--limit 100]
      --out results/tuned.json   (compare base vs tuned via src/report.py)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from huggingface_hub import InferenceClient

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_verdict(text: str) -> dict | None:
    match = JSON_RE.search(text.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def auc(pairs: list[tuple[float, int]]) -> float:
    """Pairwise AUC (handles ties). pairs = [(score, is_scam)]."""
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    wins = ties = 0
    for i, (si, yi) in enumerate(pairs):
        for sj, yj in pairs[i + 1 :]:
            if yi == yj:
                continue
            s_pos, s_neg = (si, sj) if yi == 1 else (sj, si)
            wins += s_pos > s_neg
            ties += s_pos == s_neg
    return (wins + 0.5 * ties) / (pos * neg)


def evaluate_model(client: InferenceClient, model: str, rows: list[dict]) -> dict:
    scored = []
    for row in rows:
        text = ""
        for attempt in range(3):
            try:
                text = client.chat_completion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a scam-detection assistant for Indian SMS/WhatsApp "
                                "messages. Reply with ONLY a JSON verdict: "
                                '{"risk": <0-100>, "scam_type": <family or null>, '
                                '"red_flags": [<exact phrases from the message>], "action": "<one line>"}'
                            ),
                        },
                        {"role": "user", "content": row["text"]},
                    ],
                    max_tokens=200,
                    temperature=0.0,
                ).choices[0].message.content
                break
            except Exception:
                if attempt == 2:
                    text = ""
                    time.sleep(5)
                time.sleep(2 * (attempt + 1))
        verdict = extract_verdict(text or "")
        risk = verdict.get("risk") if verdict else None
        scored.append(
            {
                "label": row["label"],
                "type": row["scam_type"],
                "verdict": verdict,
                "risk": float(risk) if isinstance(risk, (int, float)) else None,
                "red_flags_hit": (
                    sum(1 for f in verdict.get("red_flags", []) if f in row["text"])
                    if verdict
                    else 0
                ),
            }
        )
        if (len(scored)) % 25 == 0:
            print(f"  {len(scored)}/{len(rows)}", flush=True)

    pairs = [(s["risk"] or 0.0, 1 if s["label"] == "scam" else 0) for s in scored if s["risk"] is not None]
    benign = [s for s in scored if s["label"] == "benign"]
    scams = [s for s in scored if s["label"] == "scam"]
    type_f1 = _type_f1(scams)
    calibration = _calibration(pairs)
    return {
        "model": model,
        "n": len(scored),
        "metrics": {
            "auc": round(auc(pairs), 3) if pairs else None,
            "verdict_json_rate": round(sum(1 for s in scored if s["verdict"]) / len(scored), 3),
            "benign_fpr": round(
                sum(1 for s in benign if (s["risk"] or 0) >= 50) / max(len(benign), 1), 3
            ),
            "type_f1": round(type_f1, 3) if type_f1 is not None else None,
            "red_flags_precision": round(
                sum(s["red_flags_hit"] for s in scams)
                / max(sum(len(s["verdict"].get("red_flags", [])) for s in scams if s["verdict"]), 1),
                3,
            ),
        },
        "calibration": calibration,
        "samples": scored[:20],
    }


def _type_f1(scams: list[dict]) -> float | None:
    """Macro-F1 over scam families (null prediction = miss)."""
    by_pred: dict[str, int] = {}
    by_gold: dict[str, int] = {}
    hits: dict[str, int] = {}
    for s in scams:
        pred = (s["verdict"] or {}).get("scam_type") or "__none__"
        gold = s["type"] or "__none__"
        by_pred[pred] = by_pred.get(pred, 0) + 1
        by_gold[gold] = by_gold.get(gold, 0) + 1
        if pred == gold:
            hits[gold] = hits.get(gold, 0) + 1
    f1s = []
    for label in set(by_gold) | set(by_pred):
        tp = hits.get(label, 0)
        precision = tp / by_pred[label] if by_pred.get(label) else 0.0
        recall = tp / by_gold[label] if by_gold.get(label) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s) if f1s else None


def _calibration(pairs: list[tuple[float, int]]) -> list[dict]:
    buckets = [(0, 25), (25, 50), (50, 75), (75, 101)]
    out = []
    for lo, hi in buckets:
        grp = [(r, y) for r, y in pairs if lo <= r < hi]
        if grp:
            out.append(
                {
                    "bucket": f"{lo}-{hi - 1}",
                    "predicted_mean_risk": round(sum(r for r, _ in grp) / len(grp), 1),
                    "actual_scam_rate": round(sum(y for _, y in grp) / len(grp), 3),
                    "n": len(grp),
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")][: args.limit]
    client = InferenceClient()
    print(f"evaluating {args.model} on {len(rows)} messages…")
    result = evaluate_model(client, args.model, rows)
    out = Path(args.out or f"results/{args.model.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["metrics"], indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
