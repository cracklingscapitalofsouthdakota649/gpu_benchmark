   # tests/test_data_preprocessing.py
import pytest
import json
import allure
import time # ⬅️ ADDED for benchmarking

@allure.feature("Data Pipeline Benchmarks")
@allure.story("Simple CPU Data Preprocessing")
@pytest.mark.preprocess
@pytest.mark.cpu
def test_data_preprocessing():
    # Setup a large data set for a meaningful benchmark
    data_size = 1_000_000
    data = list(range(data_size))

    # Core Logic Benchmark: List comprehension
    start = time.perf_counter()
    processed = [i * 2 for i in data]
    duration = time.perf_counter() - start

    # Calculate Metric: Throughput (items/sec)
    throughput = round(data_size / duration, 2)

    # Attach the final metric for Allure trending (TEXT format)
    allure.attach(f"{throughput}", name="Preprocessing Throughput (items/sec)", attachment_type=allure.attachment_type.TEXT) # ⬅️ ADDED

    # Attach raw data (optional, for visualization)
    # allure.attach(json.dumps(processed[:10]), "Processed Data Snippet", allure.attachment_type.JSON) 
    
    # Simple check
    assert len(processed) == data_size   