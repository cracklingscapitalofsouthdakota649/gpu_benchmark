# supports/telemetry_hook.py
"""
Pytest plugin: automatically collect telemetry per test.
"""

import pytest
import os
from supports.telemetry_collector import TelemetryCollector, STOP_EVENT

COLLECTORS = {}

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Start telemetry collector before each test."""
    test_name = item.name
    build_number = os.getenv("BUILD_NUMBER", "manual")
    dest_dir = os.path.join("allure-results", build_number)
    collector = TelemetryCollector(test_name, dest_dir, duration=0, interval=0.5)
    collector.start()
    COLLECTORS[test_name] = collector


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Stop telemetry collector after each test."""
    test_name = item.name
    collector = COLLECTORS.pop(test_name, None)
    if collector:
        try:
            collector.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop telemetry for {test_name}: {e}")
    STOP_EVENT.clear()  # reset for next test


def pytest_sessionfinish(session, exitstatus):
    """Ensure all collectors stopped at session end."""
    for name, collector in list(COLLECTORS.items()):
        try:
            collector.stop()
        except Exception:
            pass
    COLLECTORS.clear()
    STOP_EVENT.set()
