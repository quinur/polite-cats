"""Tests for SHAP explainer."""

from __future__ import annotations

import types

import pandas as pd

import src.feature_importance_explainer as explainer


class DummyExplainer:
    def __init__(self, model, x):
        self.model = model
        self.x = x

    def __call__(self, x):
        return x


class DummyModel:
    """Minimal model stub with predict_proba so explain_model uses the proba branch."""
    def predict_proba(self, x):
        import numpy as np
        return np.column_stack([np.zeros(len(x)), np.ones(len(x))])


def test_explain_model_creates_plot(tmp_path, monkeypatch):
    def dummy_summary_plot(values, x, show=False):
        return None

    dummy_shap = types.SimpleNamespace(Explainer=DummyExplainer, summary_plot=dummy_summary_plot)
    monkeypatch.setattr(explainer, "shap", dummy_shap)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    output = explainer.explain_model(DummyModel(), df, tmp_path / "shap.png")
    assert (tmp_path / "shap.png").exists()
    assert output.endswith("shap.png")