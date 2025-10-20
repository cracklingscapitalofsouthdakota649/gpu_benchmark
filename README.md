# 🤖 GPU/CPU Benchmark Suite

A robust benchmarking framework for evaluating GPU and CPU performance using PyTorch, pytest, and Allure reporting.  
Supports NVIDIA, AMD, Intel, DirectML, and CPU-only execution, with performance metrics and interactive dashboards.

---

## 👤 Author & Contact  
**Author:** Bang Thien Nguyen  
**Contact:** ontario1998@gmail.com  

---

## 💡 Project Overview  
This framework implements **GPU/CPU performance benchmarking** using `pytest` and `pytest-benchmark`.  
It automatically detects available accelerators, measures inference throughput, GPU/CPU utilization, and memory usage, and produces **interactive Allure reports** for analysis.

| **Component** | **Technology** | **Role** |
|---------------|----------------|----------|
| Test Runner | **pytest** | Executes benchmark and stress tests. |
| Performance Metrics | **pytest-benchmark / SystemMetrics** | Measures FPS, CPU/GPU utilization, memory usage. |
| GPU Detection | **gpu_check.py** | Detects NVIDIA CUDA, AMD ROCm, Intel GPU, DirectML, or CPU fallback. |
| Reporting | **Allure** | Generates professional, interactive HTML dashboards with charts. |

---

## 🚀 Getting Started  

### 🔧 Prerequisites  
- 🐍 Python 3.10+ (recommended)  
- 📈 Optional: Allure command-line tool for report viewing  
- 💻 Windows or Linux system with GPU support (optional for CPU-only fallback)  

### ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/luckyjoy/gpu_benchmark.git
cd gpu_benchmark
```

Run the setup script to create a virtual environment, install dependencies, and detect GPU:  
```bash
python gpu_benchmark.py
```

The script will:  
- Create `venv310` if missing  
- Detect available GPU or fall back to CPU  
- Install all required packages including PyTorch  
- Run benchmark tests and store results in `allure-results/`  

---

## 🐳 Dockerized Execution (Optional)

Ensure consistent results across systems by running inside Docker.

### 🧱 Docker Image  
Image: **`gpu-benchmark:latest`** — includes:  
- Python 3.10 environment  
- PyTorch & required packages preinstalled  
- Allure CLI for reporting  
- `/app` as working directory  

### ▶️ Run Tests via Script  

**Script:** `run_docker.bat` (Windows)  
**Workflow:**  

| **Step** | **Description** |
|-----------|-----------------|
| 1️⃣ Check Docker | Verifies Docker Desktop is running. |
| 2️⃣ Clean Up | Deletes previous `allure-results` and `.benchmarks`. |
| 3️⃣ Build / Pull | Builds or updates Docker image. |
| 4️⃣ Execute Tests | Runs GPU/CPU benchmark suite. |
| 5️⃣ Generate Report | Produces Allure HTML output. |
| 6️⃣ Serve Report | Opens Allure dashboard locally. |

Command to execute:  
```bash
run_docker.bat
```

---

## 🌳 Framework Architecture  

```
gpu_benchmark/
├─ README.md
├─ gpu_benchmark.py           # Setup & execution script
├─ pytest.ini                 # Pytest configuration
├─ supports/                  # GPU detection & utility scripts
│  └─ gpu_check.py
├─ scripts/
│  ├─ plot_gpu_metrics.py     # Generate charts for Allure
│  └─ system_metrics.py       # Capture CPU/GPU metrics
├─ tests/                     # Benchmark test cases
│  ├─ test_data_preprocessing.py
│  ├─ test_gpu_stress.py
│  ├─ test_inference_load.py
│  └─ test_idle_baseline.py
├─ venv310/                   # Virtual environment (auto-created)
├─ allure-results/            # Benchmark reports
└─ .benchmarks/               # Pytest-benchmark history
```

---

## 🏷️ Test Tags & Execution  

| **Tag** | **Focus Area** | **Description** |
|----------|----------------|-----------------|
| `gpu` | GPU Benchmark | Tests running on CUDA/ROCm/DirectML/Intel GPU. |
| `cpu` | CPU Benchmark | Tests running on CPU fallback. |
| `stress` | GPU Stress | Heavy-load GPU endurance tests. |
| `benchmark` | Performance | FPS, utilization, memory measurement. |

### 🧪 Run CMD Tests Locally  

| **Mode** | **Command** |
|-----------|-------------|
| Run All GPU Tests | `pytest -m gpu --alluredir=allure-results -v` |
| Run All CPU Tests | `pytest -m cpu --alluredir=allure-results -v` |
| Run Specific Benchmark | `pytest -m "benchmark and gpu"` |

---

## 📊 Professional Test Reporting  

### 1️⃣ **Interactive Allure Report (Recommended)**  
```bash
pytest -m gpu --alluredir=allure-results
allure serve allure-results
```

📸 *Preview of GPU Metrics Dashboard:*  

![Allure Overview Report](docs/screenshots/allure_gpu_overview.png)  
*Shows FPS, GPU utilization, and memory usage over time.*

![Allure Pytest Suites Report](docs/screenshots/allure_gpu_suites.png)  
*Detailed view per test with step-by-step metrics.*

📸 *CPU Metrics Preview:*  

![Allure CPU Overview](docs/screenshots/allure_cpu_overview.png)  
*Tracks CPU utilization, memory usage, and benchmark throughput.*

> Opens an interactive HTML dashboard with detailed execution insights.

### 2️⃣ **Static HTML Report (Optional)**  
```bash
pytest --html=reports/report.html --self-contained-html
```

---

## ⚙️ CI/CD Integration

| System | Description |
|--------|-------------|
| Jenkins / GitHub Actions | Automates test execution and report generation |
| Docker | Guarantees repeatable benchmark environments |
| Allure | Produces professional dashboards for CI/CD pipelines |

---

## 🤝 Contributing Guidelines

1. Fork the repository  
2. Create a feature branch  
3. Implement new tests, benchmarks, or reporting features  
4. Run `pytest -v` locally and verify results  
5. Submit a Pull Request with a clear description  

**Code Style:**  
- Follow **PEP8** conventions  
- Use **pytest markers** consistently  
- Ensure **Allure reports** generate without errors  
- Document new metrics or tests in **Allure charts**  

---

## 🪪 License

Released under the **MIT License** — free to use, modify, and distribute.

---

📬 *Contact:* [ontario1998@gmail.com](mailto:ontario1998@gmail.com)  

> _“Measure performance before you optimize — know your hardware before you test your code.”_
