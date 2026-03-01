"""Feature importance explanations using SHAP values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import shap
except (ImportError, OSError):  # pragma: no cover - optional or broken dependency
    shap = None


def explain_model(model: Any, x: pd.DataFrame, output_path: str | Path) -> str:
    """Generate SHAP summary plot for a trained model.

    Args:
        model: Trained model or pipeline.
        x: Feature DataFrame.
        output_path: Path to save the plot.

    Returns:
        Saved plot path.
    """
    if shap is None:
        raise ImportError("shap is required to explain the model.")
    
    # Use KernelExplainer for stable results across different model types
    # or use a prediction function directly
    if hasattr(model, "predict_proba"):
        f = lambda x_input: model.predict_proba(x_input)[:, 1]
    else:
        f = model.predict
        
    explainer = shap.Explainer(f, x)
    shap_values = explainer(x)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)