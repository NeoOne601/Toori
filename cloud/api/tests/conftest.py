import sys

import pytest


if sys.version_info[:2] != (3, 11):
    pytest.skip(
        "cloud/api/tests require Python 3.11 (runtime is intentionally guarded on 3.11)",
        allow_module_level=True,
    )

def pytest_collection_modifyitems(config, items):
    """Add timeout markers to known-slow test suites."""
    for item in items:
        if "test_worker_pool" in item.nodeid:
            item.add_marker(pytest.mark.timeout(45))
        elif "test_immersive_engine" in item.nodeid:
            item.add_marker(pytest.mark.timeout(30))
        elif "test_existing_toori_functionality_not_broken" in item.nodeid:
            item.add_marker(pytest.mark.timeout(120))
