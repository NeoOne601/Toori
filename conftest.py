"""
conftest.py  (project root — /Users/macuser/toori/conftest.py)

Sets TOORI_VJEPA2_ENV=test before any module import.
Auto-skips @pytest.mark.vjepa2 tests in the default suite.
"""
import os
import pytest

os.environ.setdefault("TOORI_VJEPA2_ENV",    "test")
os.environ.setdefault("TOORI_VJEPA2_FRAMES", "4")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "vjepa2: tests loading the real V-JEPA 2 model. "
        "Excluded from default suite. Run with: pytest -m vjepa2",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests expected to take > 10 seconds.",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip vjepa2-marked tests unless -m vjepa2 is explicitly set."""
    marker_expr = config.getoption("-m", default="")
    run_vjepa2 = "vjepa2" in marker_expr

    skip_vjepa2 = pytest.mark.skip(
        reason=(
            "V-JEPA 2 tests excluded from default suite to prevent OOM. "
            "Run separately: TOORI_VJEPA2_ENV=test TOORI_VJEPA2_FRAMES=4 "
            "pytest -m vjepa2 -v cloud/jepa_service/tests/test_world_model_predictor.py"
        )
    )

    for item in items:
        if "vjepa2" in item.keywords and not run_vjepa2:
            item.add_marker(skip_vjepa2)
