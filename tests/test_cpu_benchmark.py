# tests\test_cpu_benchmark.py

"""
CPU Benchmark Suite
-------------------
These tests benchmark CPU-intensive, memory-bound, and I/O-like operations.
They use Python built-ins only — no GPU or torch dependency.
All tests emit Allure features and stories for reporting.
"""
import time
import pytest
import allure

@allure.feature("CPU Benchmark")
@allure.story("Sorting Performance")
@allure.severity(allure.severity_level.NORMAL)
@pytest.mark.cpu
def test_cpu_sorting_benchmark():
    data = [x for x in range(1000000, 0, -1)]
    start = time.time()
    sorted_data = sorted(data)
    duration = time.time() - start

    allure.attach(str(duration), "Execution Time (s)", allure.attachment_type.TEXT)
    assert sorted_data[0] == 1
    assert duration < 0.01  # should sort under 1s on most CPUs


@allure.feature("CPU Benchmark")
@allure.story("Prime Number Calculation")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.cpu
def test_cpu_prime_benchmark():
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    start = time.time()
    primes = [n for n in range(2, 5000) if is_prime(n)]
    duration = time.time() - start

    allure.attach(str(len(primes)), "Number of Primes", allure.attachment_type.TEXT)
    allure.attach(str(duration), "Execution Time (s)", allure.attachment_type.TEXT)
    assert len(primes) > 600
    assert duration < 0.01


@allure.feature("CPU Benchmark")
@allure.story("Fibonacci Sequence Calculation")
@allure.severity(allure.severity_level.MINOR)
@pytest.mark.cpu
def test_cpu_fibonacci_benchmark():
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    start = time.time()
    result = fib(30)
    duration = time.time() - start

    allure.attach(str(result), "Fib(30)", allure.attachment_type.TEXT)
    allure.attach(str(duration), "Execution Time (s)", allure.attachment_type.TEXT)
    assert result == 832040
    assert duration < 0.1


@allure.feature("CPU Benchmark")
@allure.story("Matrix Arithmetic Simulation")
@allure.severity(allure.severity_level.NORMAL)
@pytest.mark.cpu
def test_cpu_matrix_arithmetic():
    size = 60
    matrix = [[i * j for j in range(size)] for i in range(size)]

    start = time.time()
    result = 0
    for i in range(size):
        for j in range(size):
            result += matrix[i][j] * (i + j)
    duration = time.time() - start

    allure.attach(str(result), "Matrix Result", allure.attachment_type.TEXT)
    allure.attach(str(duration), "Execution Time (s)", allure.attachment_type.TEXT)
    assert result > 0
    assert duration < 0.1
