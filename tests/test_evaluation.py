"""Tests for evaluation module."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.evaluation import compute_metrics, compute_dialectal_bias


def test_compute_metrics_perfect():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]
    m = compute_metrics(np.array(y_true), np.array(y_pred))
    assert m["accuracy"]  == 1.0
    assert m["f1"]        == 1.0
    assert m["precision"] == 1.0
    assert m["recall"]    == 1.0
    assert m["fpr"]       == 0.0


def test_compute_metrics_random():
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]
    m = compute_metrics(np.array(y_true), np.array(y_pred))
    assert 0 <= m["accuracy"]  <= 1
    assert 0 <= m["f1"]        <= 1
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"]    <= 1
    assert 0 <= m["fpr"]       <= 1


def test_dialectal_bias_improvement():
    """DialectLex should show improvement when predictions are better."""
    base = {
        "Mexico": {"y_true": [1,0,1,1,0], "y_pred": [0,1,1,0,0]},
    }
    dlx = {
        "Mexico": {"y_true": [1,0,1,1,0], "y_pred": [1,0,1,1,0]},  # perfect
    }
    df = compute_dialectal_bias(base, dlx)
    assert df.loc[df["variante"]=="Mexico", "delta_f1"].values[0] > 0
    assert df.loc[df["variante"]=="Mexico", "improved"].values[0]
