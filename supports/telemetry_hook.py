# supports/telemetry_hook.py
import os
import threading
import pytest
import allure
from supports.telemetry_collector import TelemetryCollector

RESULTS_DIR = os.getenv("ALLURE_RESULTS", "allure-results")
_collectors = {}
_lock = threading.Lock()


def _attach_file(path, name, mime):
    try:
        with open(path, "rb") as f:
            data = f.read()
        allure.attach(data, name=name, attachment_type=mime)
        return True
    except Exception:
        print(f"[WARN] Could not attach {name}")
        return False


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Start telemetry collection per test module."""
    mod = getattr(item, "module", None)
    if not mod:
        return
    mod_name = os.path.splitext(os.path.basename(getattr(mod, "__file__", "unknown")))[0]
    with _lock:
        if mod_name not in _collectors:
            c = TelemetryCollector(module_name=mod_name, sample_interval=1.0)
            c.start()
            _collectors[mod_name] = c
            print(f"[INFO] Started telemetry for {mod_name}")


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Stop telemetry at end of module and attach data to Allure."""
    mod = getattr(item, "module", None)
    if not mod:
        return
    mod_name = os.path.splitext(os.path.basename(getattr(mod, "__file__", "unknown")))[0]

    # Determine if this is the last test in the module
    session_items = [i for i in item.session.items if getattr(i, "module", None) == mod]
    current_index = session_items.index(item)
    is_last = current_index == len(session_items) - 1

    if is_last:
        with _lock:
            c = _collectors.pop(mod_name, None)
        if not c:
            return
        data = c.stop()
        json_path = c.save_json(RESULTS_DIR)
        png_path = c.plot_png(RESULTS_DIR)
        print(f"[INFO] Saved telemetry for module {mod_name} ({len(data)} samples)")

        _attach_file(json_path, f"Telemetry {mod_name} (JSON)", allure.attachment_type.JSON)
        if png_path:
            _attach_file(png_path, f"Telemetry {mod_name} (Chart)", allure.attachment_type.PNG)
