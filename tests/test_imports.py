"""Optional-dependency import smoke tests.

These are intentionally non-pytest test functions — they just print
whether optional packages are available on this system.
"""

try:
    import streamlit  # noqa: F401
    print("streamlit imported")
except (ImportError, OSError) as e:
    print(f"streamlit import failed: {e}")

try:
    import shap  # noqa: F401
    print("shap imported")
except (ImportError, OSError) as e:
    print(f"shap import failed: {e}")

try:
    from fpdf import FPDF  # noqa: F401
    print("fpdf imported")
except (ImportError, OSError) as e:
    print(f"fpdf import failed: {e}")

try:
    from src.dashboard import run_dashboard  # noqa: F401
    print("src.dashboard imported")
except (ImportError, OSError) as e:
    print(f"src.dashboard import failed: {e}")
