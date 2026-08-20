"""
hybrid.py - Hybrid Detection Router (ML + LLM Judge)
This is the architecture piece that makes the system a genuine hybrid
rather than a single classifier: the embedding classifier (fast, <5ms)
gates every request, and the LLM judge (src/llm_judge.py) is only called
for the ambiguous middle band.

    User Prompt
         |
    ML Classifier (embedding-based, <5ms)   <- src/classifier_v2.py
         |
    Risk Score
         |
      < LOW_THRESHOLD   -> ALLOW directly
      LOW..HIGH         -> LLM Judge (secondary review)   <- src/llm_judge.py
      > HIGH_THRESHOLD  -> BLOCK directly
         |
    ALLOW / REVIEW / BLOCK

Usage:
    python src/hybrid.py                          # run on full dataset, print metrics
    python src/hybrid.py --predict                # interactive mode
    python src/hybrid.py --text "ignore all rules" # single prediction
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from src.classifier_v2 import load_model as load_ml_model, predict as ml_predict

# NOTE: src.llm_judge (and therefore torch/transformers) is imported lazily
# inside route(), not here at module level. This lets src.hybrid be imported
# — and its routing logic tested — on a machine with no GPU and no
# torch/transformers installed. Same pattern as src/proxy.py.

# ── Config ───────────────────────────────────────────────────────────────────
LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.75

DATASET_PATH = Path("data/processed/dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
METRICS_PATH = MODEL_DIR / "hybrid_metrics.json"
RESULTS_PATH = Path("results") / "full_evaluation_results.csv"


# ── Router ───────────────────────────────────────────────────────────────────
def route(text: str, embed_model, clf, low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD,
          judge_fn=None) -> dict:
    """
    Routes a single prompt through the hybrid pipeline.

    `judge_fn` is optional dependency injection for testing: pass a mock
    callable (text -> dict with a "verdict" key) to exercise the borderline
    branch without needing torch/transformers/a GPU. When omitted, the real
    LLM judge (src/llm_judge.py) is imported lazily and used.

    Returns a dict with verdict, source ("ML" or "LLM"), is_injection, and
    supporting detail (ml_risk_score always included; llm fields only when
    the LLM judge was actually invoked).
    """
    ml_result = ml_predict(text, embed_model, clf)
    risk_score = ml_result["confidence"]

    if risk_score < low:
        return {
            "text": text,
            "verdict": "ALLOW",
            "source": "ML",
            "is_injection": False,
            "ml_risk_score": risk_score,
            "reasoning": f"ML risk score {risk_score:.2f} below {low} — allowed directly.",
        }

    if risk_score > high:
        return {
            "text": text,
            "verdict": "BLOCK",
            "source": "ML",
            "is_injection": True,
            "ml_risk_score": risk_score,
            "reasoning": f"ML risk score {risk_score:.2f} above {high} — blocked directly.",
        }

    # Borderline — escalate to the LLM judge (imported lazily unless injected)
    if judge_fn is None:
        from src.llm_judge import judge as judge_fn
    judge_result = judge_fn(text)
    judge_result["source"] = "LLM"
    judge_result["ml_risk_score"] = risk_score
    return judge_result


# ── Batch evaluation ─────────────────────────────────────────────────────────
def evaluate(low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD) -> dict:
    """
    Runs the hybrid router over the full dataset.

    ALLOW / REVIEW / BLOCK are three distinct actions, not a binary — REVIEW
    is a genuinely separate outcome (flagged for human review), so it is NOT
    folded into "injection" or "not injection" for scoring purposes. Two
    numbers are reported instead:

    1. F1 on the *resolved* cases (ALLOW vs BLOCK only) — the accuracy of
       every automatic decision the system actually made.
    2. REVIEW-bucket stats — how many prompts were escalated to a human,
       and what fraction of those were actually injections vs benign. A
       well-calibrated REVIEW bucket should be enriched with genuinely
       ambiguous cases, not dominated by one label.
    """
    print("=" * 52)
    print("  Hybrid Router (ML + LLM Judge) — Evaluation")
    print("=" * 52)

    df = pd.read_csv(DATASET_PATH)
    embed_model, clf = load_ml_model()

    n_llm_calls = 0
    rows = []
    for i, row in df.iterrows():
        result = route(row["text"], embed_model, clf, low, high)
        if result["source"] == "LLM":
            n_llm_calls += 1
        rows.append({
            "text": row["text"],
            "category": row["category"],
            "label": row["label"],
            "hybrid_verdict": result["verdict"],   # ALLOW / REVIEW / BLOCK
            "hybrid_source": result["source"],
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(df)}] processed — {n_llm_calls} LLM calls so far")

    df_hybrid = pd.DataFrame(rows)

    resolved = df_hybrid[df_hybrid["hybrid_verdict"] != "REVIEW"].copy()
    review = df_hybrid[df_hybrid["hybrid_verdict"] == "REVIEW"].copy()

    y_true = (resolved["label"] == "injection").astype(int)
    y_pred = (resolved["hybrid_verdict"] == "BLOCK").astype(int)

    f1 = f1_score(y_true, y_pred) if len(resolved) else 0.0
    cm = confusion_matrix(y_true, y_pred) if len(resolved) else [[0, 0], [0, 0]]
    tn, fp, fn, tp = (cm.ravel() if len(resolved) else [0, 0, 0, 0])

    review_injection_count = int((review["label"] == "injection").sum())
    review_benign_count = int((review["label"] == "benign").sum())

    metrics = {
        "model": "Hybrid (embedding classifier gate + Mistral-7B LLM judge)",
        "low_threshold": low,
        "high_threshold": high,
        "dataset_size": len(df),
        "resolved_cases": len(resolved),
        "f1_resolved": round(f1, 4),
        "confusion_matrix_resolved": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "review_bucket": {
            "count": len(review),
            "review_rate": round(len(review) / len(df), 4),
            "actual_injection": review_injection_count,
            "actual_benign": review_benign_count,
        },
        "llm_calls": n_llm_calls,
        "llm_call_rate": round(n_llm_calls / len(df), 4),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    df_hybrid.to_csv(RESULTS_PATH, index=False)

    print(f"\n── Metrics ──────────────────────────────────────")
    print(f"  F1 (resolved cases) : {f1:.4f}  ({len(resolved)}/{len(df)} auto-decided)")
    print(f"  REVIEW bucket        : {len(review)} prompts ({len(review) / len(df):.1%}) "
          f"— {review_injection_count} actual injection, {review_benign_count} actual benign")
    print(f"  LLM calls            : {n_llm_calls}/{len(df)} ({n_llm_calls / len(df):.1%})")
    print(f"\n[INFO] Metrics saved -> {METRICS_PATH}")
    print(f"[INFO] Per-example results saved -> {RESULTS_PATH}")
    return metrics


# ── Interactive ──────────────────────────────────────────────────────────────
def interactive_route():
    embed_model, clf = load_ml_model()
    print("\n── Interactive Prediction (Hybrid Router) ────────")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        text = input("Prompt> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = route(text, embed_model, clf)
        icon = {"ALLOW": "✅", "REVIEW": "🟡", "BLOCK": "🚨"}[result["verdict"]]
        print(f"  {icon} {result['verdict']}  (source={result['source']}, "
              f"ml_risk={result['ml_risk_score']:.2f})")
        if result["source"] == "LLM":
            print(f"     reason: {result.get('reasoning', '')}")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="interactive mode")
    parser.add_argument("--text", type=str, default=None, help="classify a single prompt and exit")
    parser.add_argument("--low", type=float, default=LOW_THRESHOLD)
    parser.add_argument("--high", type=float, default=HIGH_THRESHOLD)
    args = parser.parse_args()

    if args.text:
        embed_model, clf = load_ml_model()
        result = route(args.text, embed_model, clf, args.low, args.high)
        print(json.dumps(result, indent=2, default=str))
    elif args.predict:
        interactive_route()
    else:
        evaluate(args.low, args.high)
