"""
eval_baseline.py - Evaluation script for the baseline TF-IDF classifier.
Generates confusion matrix, error analysis, and saves results.

Usage:
    python src/eval_baseline.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH  = Path("data/processed/dataset.csv")
MODEL_PATH    = Path("models/logreg_classifier.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
METRICS_PATH  = Path("models/baseline_metrics.json")
ERRORS_PATH   = Path("models/error_analysis.json")


def load_artifacts():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    return vectorizer, clf


def evaluate():
    print("=" * 50)
    print("  Baseline Classifier — Full Evaluation")
    print("=" * 50)

    df = pd.read_csv(DATASET_PATH)
    X = df["text"].str.strip().str.lower()
    y = (df["label"] == "injection").astype(int)

    vectorizer, clf = load_artifacts()
    X_vec = vectorizer.transform(X)

    # ── Cross-validation ─────────────────────────────────────────────────────
    print("\n── 5-Fold Cross Validation ─────────────────────")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_vec, y, cv=cv, scoring="f1")
    print(f"  F1 per fold : {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Full dataset predictions ──────────────────────────────────────────────
    y_pred = clf.predict(X_vec)
    y_prob = clf.predict_proba(X_vec)[:, 1]

    print("\n── Classification Report ───────────────────────")
    print(classification_report(y, y_pred, target_names=["benign", "injection"]))

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn)
    auc = roc_auc_score(y, y_prob)

    print("── Key Metrics ─────────────────────────────────")
    print(f"  ROC-AUC            : {auc:.4f}")
    print(f"  False Positive Rate: {fpr_val:.4f}  ({fp} benign flagged as injection)")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}  ({fn} injections missed)")
    print(f"  Confusion Matrix   :")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    # ── Error analysis ────────────────────────────────────────────────────────
    df["predicted"] = ["injection" if p == 1 else "benign" for p in y_pred]
    df["confidence"] = y_prob

    false_positives = df[(df["label"] == "benign") & (df["predicted"] == "injection")]
    false_negatives = df[(df["label"] == "injection") & (df["predicted"] == "benign")]

    print(f"\n── Error Analysis ──────────────────────────────")
    print(f"  False Positives (benign flagged): {len(false_positives)}")
    for _, row in false_positives.iterrows():
        print(f"    [{row['confidence']:.3f}] {row['text'][:70]}")

    print(f"\n  False Negatives (injections missed): {len(false_negatives)}")
    for _, row in false_negatives.iterrows():
        print(f"    [{row['confidence']:.3f}] {row['text'][:70]}")

    # ── Threshold sensitivity ─────────────────────────────────────────────────
    print(f"\n── Threshold Sensitivity ───────────────────────")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (y_prob >= threshold).astype(int)
        cm_t = confusion_matrix(y, preds)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        fpr_t = fp_t / (fp_t + tn_t)
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0
        print(f"  threshold={threshold}  FPR={fpr_t:.3f}  FNR={fnr_t:.3f}  TP={tp_t}  FP={fp_t}")

    # ── Category breakdown ────────────────────────────────────────────────────
    print(f"\n── Accuracy by Category ────────────────────────")
    injection_df = df[df["label"] == "injection"].copy()
    injection_df["correct"] = injection_df["predicted"] == "injection"
    cat_acc = injection_df.groupby("category")["correct"].mean().sort_values()
    for cat, acc in cat_acc.items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cat:<25} {bar} {acc:.0%}")

    # ── Save everything ───────────────────────────────────────────────────────
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_size": len(df),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fpr_val, 4),
        "false_negative_rate": round(fn / (fn + tp), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    errors = {
        "false_positives": false_positives[["text", "confidence"]].to_dict("records"),
        "false_negatives": false_negatives[["text", "confidence"]].to_dict("records"),
    }
    with open(ERRORS_PATH, "w") as f:
        json.dump(errors, f, indent=2)

    print(f"\n[INFO] Metrics saved → {METRICS_PATH}")
    print(f"[INFO] Error analysis saved → {ERRORS_PATH}")
    print("\n[DONE] Run Day 8 to upgrade to sentence-transformer embeddings.")


if __name__ == "__main__":
    evaluate()
