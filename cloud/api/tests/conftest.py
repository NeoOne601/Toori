import sys

import pytest


if sys.version_info[:2] != (3, 11):
    pytest.skip(
        "cloud/api/tests require Python 3.11 (runtime is intentionally guarded on 3.11)",
        allow_module_level=True,
    )

