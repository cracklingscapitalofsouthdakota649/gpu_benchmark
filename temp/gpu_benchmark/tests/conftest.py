# conftest.py
"""
Global pytest configuration for dynamic timeouts, cache cleanup,
and GPU memory management between tests.
"""

import gc
import pytest
import os
import torch

# Register the telemetry hook as a pytest plugin
pytest_plugins = ["supports.telemetry_hook"]

# ───────────────────────────────────────────────────────────────
# Global Test Setup / Teardown
# ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_memory_between_tests():
    """
    Automatically clears GPU and CPU caches between tests
    to ensure no cross-test interference or OOM issues.
    """
    # Before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yield  # Run the actual test

    # After test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# ───────────────────────────────────────────────────────────────
# Dynamic Timeout Handling per Marker
# ───────────────────────────────────────────────────────────────

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """
    Dynamically assign timeout values depending on test type/marker.
    Prevents global 60s limit from killing long-running benchmarks.
    """
    # Skip if test explicitly defines @pytest.mark.timeout
    if item.get_closest_marker("timeout"):
        return

    # Default timeout (seconds)
    timeout = 60  # Normal test = 1 min

    # GPU tests — may require kernel launch or sync
    if item.get_closest_marker("gpu") or item.get_closest_marker("accelerator"):
        timeout = 300  # 5 minutes

    # I/O or network accelerator tests
    elif item.get_closest_marker("io_accelerator") or item.get_closest_marker("network_io_accelerator"):
        timeout = 600  # 10 minutes

    # Stress / benchmark tests
    elif item.get_closest_marker("stress") or item.get_closest_marker("benchmark"):
        timeout = 900  # 15 minutes

    # Add timeout marker dynamically
    item.add_marker(pytest.mark.timeout(timeout, method="thread"))

# ───────────────────────────────────────────────────────────────
# Optional: Session-Level GPU Cache Clear
# ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def init_environment():
    """
    Prepare environment variables and ensure GPU is initialized cleanly.
    This runs once per pytest session.
    """
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield

    # Final cleanup after all tests
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
