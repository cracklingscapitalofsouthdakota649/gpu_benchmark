#!/usr/bin/env python3
import importlib.util, os, sys, pprint

print("=== Pre-flight Python Env ===")
print("Python:", sys.version)
print("CWD:", os.getcwd())
print("PYTHONPATH env:", os.environ.get("PYTHONPATH"))
print("supports/gpu_check.py exists?:", os.path.exists("supports/gpu_check.py"))
print("tests/device_utils.py exists?:", os.path.exists("tests/device_utils.py"))
print("find_spec('supports'):", importlib.util.find_spec("supports"))
print("find_spec('supports.gpu_check'):", importlib.util.find_spec("supports.gpu_check"))
print("sys.path:")
pprint.pprint(sys.path)
print("=== End pre-flight ===")