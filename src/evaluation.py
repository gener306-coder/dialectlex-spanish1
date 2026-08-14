"""
evaluation.py — Evaluation metrics and dialectal bias analysis.

Provides functions to compute standard classification metrics
and compare performance across dialectal variants.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from typing import Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, f1 and fpr.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "fpr":       round(fpr, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_dialectal_bias(
    results_base: dict,
    results_dialectlex: dict,
) -> pd.DataFrame:
    """
    Compare metrics between BETO base and BETO + DialectLex
    across dialectal variants and compute bias indicators.

    Parameters
    ----------
    results_base : dict
        {variant_name: {"y_true": [...], "y_pred": [...]}}
    results_dialectlex : dict
        Same structure for the DialectLex model.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-variant metrics and deltas.
    """
    rows = []
    for variant in results_base:
        yt = np.array(results_base[variant]["y_true"])
        yp_base = np.array(results_base[variant]["y_pred"])
        yp_dlx  = np.array(results_dialectlex[variant]["y_pred"])

        m_base = compute_metrics(yt, yp_base)
        m_dlx  = compute_metrics(yt, yp_dlx)

        rows.append({
            "variante":        variant,
            "n":               len(yt),
            "f1_base":         m_base["f1"],
            "f1_dialectlex":   m_dlx["f1"],
            "delta_f1":        round(m_dlx["f1"] - m_base["f1"], 4),
            "recall_base":     m_base["recall"],
            "recall_dialectlex": m_dlx["recall"],
            "delta_recall":    round(m_dlx["recall"] - m_base["recall"], 4),
            "fpr_base":        m_base["fpr"],
            "fpr_dialectlex":  m_dlx["fpr"],
        })

    df = pd.DataFrame(rows)
    df["improved"] = df["delta_f1"] > 0
    return df


def print_bias_report(df: pd.DataFrame) -> None:
    """
    Print a formatted bias analysis report.

    Parameters
    ----------
    df : pd.DataFrame
        Output of compute_dialectal_bias().
    """
    print("\n" + "=" * 65)
    print("DIALECTAL BIAS ANALYSIS")
    print("=" * 65)
    print(f"{'Variante':<20} {'n':>6} {'F1 base':>9} {'F1 DLex':>9} {'Δ F1':>8} {'Mejora':>7}")
    print("-" * 65)
    for _, row in df.iterrows():
        flag = "✓" if row["improved"] else "✗"
        print(f"{row['variante']:<20} {int(row['n']):>6} "
              f"{row['f1_base']:>9.4f} {row['f1_dialectlex']:>9.4f} "
              f"{row['delta_f1']:>+8.4f} {flag:>7}")
    print("-" * 65)
    improved = df["improved"].sum()
    total    = len(df)
    print(f"\nResumen: mejora en {improved}/{total} variantes evaluadas.")
    best = df.loc[df["delta_f1"].idxmax()]
    print(f"Mayor ganancia: {best['variante']} "
          f"(Δ F1 = {best['delta_f1']:+.4f})")


def huggingface_compute_metrics(eval_pred):
    """
    Compute metrics for HuggingFace Trainer.

    Parameters
    ----------
    eval_pred : EvalPrediction
        HuggingFace EvalPrediction object.

    Returns
    -------
    dict
        Metrics dictionary compatible with Trainer.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    m = compute_metrics(labels, predictions)
    return {
        "f1_macro":  m["f1_macro"],
        "accuracy":  m["accuracy"],
        "precision": m["precision"],
        "recall":    m["recall"],
    }
