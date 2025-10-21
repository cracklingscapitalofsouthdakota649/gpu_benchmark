# tests/test_data_preprocessing.py
import pytest
import json
import allure

@allure.feature("Data Pipeline Benchmarks") # ⬅️ ADDED
@allure.story("Simple CPU Data Preprocessing") # ⬅️ ADDED
@pytest.mark.preprocess
@pytest.mark.gpu
@pytest.mark.cpu
def test_data_preprocessing():
    data = [i for i in range(100)]
    processed = [i*2 for i in data]
    allure.attach(json.dumps(processed), "Processed Data", allure.attachment_type.JSON)
    assert sum(processed) == sum(data)*2