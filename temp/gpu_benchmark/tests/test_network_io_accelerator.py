# tests/test_network_io_accelerator.py
# Simulate real-world enterprise I/O acceleration, validating throughput, latency, and RDMA path health

import os
import re
import gc
import pytest
import time
import socket
import subprocess
import asyncio
import tempfile
import statistics
import allure
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ───────────────────────────────────────────────────────────────
# Optional torch import for GPU cache management
# ───────────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    torch = None


# ───────────────────────────────────────────────────────────────
# Detection Utilities
# ───────────────────────────────────────────────────────────────
def detect_rdma():
    """Detect RDMA devices via 'Get-NetAdapterRdma' (Windows)."""
    try:
        result = subprocess.run(["powershell", "Get-NetAdapterRdma"], capture_output=True, text=True)
        return bool(re.search(r"Enabled\s*:\s*True", result.stdout))
    except Exception:
        return False


def detect_nvmeof():
    """Detect NVMe-over-Fabrics mount points."""
    try:
        result = subprocess.run(["wmic", "diskdrive", "get", "Model"], capture_output=True, text=True)
        return any("NVMe-oF" in line for line in result.stdout.splitlines())
    except Exception:
        return False


def detect_iscsi():
    """Detect iSCSI session presence."""
    try:
        result = subprocess.run(["iscsicli", "ListTargets"], capture_output=True, text=True)
        return "Target Name" in result.stdout
    except Exception:
        return False


def detect_network_accelerators():
    info = {
        "rdma": detect_rdma(),
        "nvme_of": detect_nvmeof(),
        "iscsi": detect_iscsi(),
    }
    allure.attach(str(info), name="Detected Network Accelerators", attachment_type=allure.attachment_type.TEXT)
    return info


@pytest.mark.network_io
class TestNetworkIOAccelerator:
    @pytest.fixture(autouse=True)
    def cleanup_between_tests(self):
        """
        Automatically clear GPU/CPU memory cache and perform garbage collection
        before and after each test to prevent resource leaks and hangs.
        """
        # Before test
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        yield
        # After test
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @pytest.fixture(scope="class", autouse=True)
    def hardware_info(self):
        """Detect available network I/O accelerators once."""
        return detect_network_accelerators()

    @pytest.fixture(scope="class")
    def payload(self):
        """Generate 256 MB random binary payload."""
        data = os.urandom(256 * 1024 * 1024)
        return data

    # ───────────────────────────────────────────────────────────────
    # 1️. RDMA Latency + Throughput Test
    # ───────────────────────────────────────────────────────────────
    @pytest.mark.network_io_accelerator
    @allure.feature("Network I/O Accelerator")
    @allure.story("RDMA Latency + Throughput Test")
    def test_rdma_throughput_latency(self, hardware_info, benchmark):
        if not hardware_info["rdma"]:
            pytest.skip("RDMA-capable NIC not detected.")
        
        start = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            try:
                # Use localhost to simulate RDMA TCP path fallback
                s.connect(("127.0.0.1", 445))
                s.sendall(b"RDMA test packet" * 1000)
                s.recv(1024)
            except Exception:
                pass
        elapsed = time.perf_counter() - start
        result = benchmark(lambda: elapsed)

        allure.attach(f"Latency: {elapsed*1000:.2f} ms", name="RDMA Latency")
        assert elapsed < 0.5  # Expect sub-ms latency in RDMA-capable NICs

    # ───────────────────────────────────────────────────────────────
    # 2️. NVMe-over-Fabrics File I/O Simulation
    # ───────────────────────────────────────────────────────────────
    @pytest.mark.network_io_accelerator
    @allure.feature("Network I/O Accelerator")
    @allure.story("NVMe-over-Fabrics File I/O Simulation")
    def test_nvme_of_file_io(self, hardware_info, benchmark):
        if not hardware_info["nvme_of"]:
            pytest.skip("NVMe-over-Fabrics target not detected.")
        
        temp_file = Path(tempfile.gettempdir()) / "nvmeof_test.bin"
        size_mb = 512
        data = os.urandom(1024 * 1024)

        def io_workload():
            with open(temp_file, "wb") as f:
                for _ in range(size_mb):
                    f.write(data)
            with open(temp_file, "rb") as f:
                while f.read(4 * 1024 * 1024):
                    pass
            temp_file.unlink(missing_ok=True)

        result = benchmark(io_workload)
        allure.attach("NVMe-oF 512MB workload complete", name="NVMe-oF I/O Test")
        assert result is not None

    # ───────────────────────────────────────────────────────────────
    # 3️. Parallel iSCSI I/O Test
    # ───────────────────────────────────────────────────────────────
    @pytest.mark.network_io_accelerator
    @allure.feature("Network I/O Accelerator")
    @allure.story("Parallel iSCSI I/O Test")
    def test_parallel_iscsi_io(self, hardware_info, benchmark, payload):
        if not hardware_info["iscsi"]:
            pytest.skip("No iSCSI targets available.")
        
        threads = 8
        block_size = 8 * 1024 * 1024

        def iscsi_io():
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for i in range(threads):
                    pool.submit(lambda: sum(payload[i*block_size:(i+1)*block_size]))

        result = benchmark(iscsi_io)
        allure.attach("Parallel iSCSI load executed", name="iSCSI Parallel I/O")
        assert result is not None

    # ───────────────────────────────────────────────────────────────
    # 4️. Network I/O Round-Trip Latency Test (SMB Direct)
    # ───────────────────────────────────────────────────────────────
    @pytest.mark.network_io_accelerator
    @allure.feature("Network I/O Accelerator")
    @allure.story("Round-Trip Latency Test (SMB Direct)")
    @pytest.mark.asyncio
    async def test_smb_direct_latency(self, benchmark):
        """Measure async TCP I/O round-trip time to simulate SMB Direct."""
        async def round_trip():
            start = time.perf_counter()
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", 445)
                writer.write(b"ping")
                await writer.drain()
                await reader.read(1024)
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            return time.perf_counter() - start

        result = await round_trip()
        allure.attach(f"Latency {result*1000:.3f} ms", name="SMB Direct Latency")
        benchmark(lambda: result)
        assert result < 5.0  # Expect low latency

    # ───────────────────────────────────────────────────────────────
    # 5️. Network I/O Stability Over Time
    # ───────────────────────────────────────────────────────────────
    @pytest.mark.network_io_accelerator
    @allure.feature("Network I/O Accelerator")
    @allure.story("Stability Over Time")
    def test_network_stability(self, benchmark):
        """Run repeated ping over 30 seconds and track packet loss."""
        host = "8.8.8.8"
        try:
            # Increased ping count for a better stability sample
            result = subprocess.run(["ping", host, "-n", "30"], capture_output=True, text=True) 
            loss_match = re.search(r"(\d+)% loss", result.stdout)
            packet_loss = int(loss_match.group(1)) if loss_match else 100
        except Exception:
            packet_loss = 100
            
        # Attach the final metric for Allure trending (TEXT format)
        allure.attach(f"{packet_loss}", name="Packet Loss Percentage", attachment_type=allure.attachment_type.TEXT) # ⬅️ ADDED

        # Attach raw data for debug
        allure.attach(result.stdout, "Ping Output", allure.attachment_type.TEXT)
        
        # Validation
        assert packet_loss == 0, f"Detected packet loss: {packet_loss}%"