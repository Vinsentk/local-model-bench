from __future__ import annotations

import ctypes
import subprocess
import threading
import time


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = [resource_snapshot()]
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.samples.append(resource_snapshot())
        return summarize_samples(self.samples)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(resource_snapshot())


def resource_snapshot() -> dict[str, float]:
    cpu_percent = _cpu_load_percent()
    ram_percent, ram_used_gb, ram_total_gb = _memory_status()
    gpu_util, vram_used_gb, vram_total_gb = _nvidia_smi()
    return {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_percent": gpu_util,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
    }


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    first = samples[0]
    summary: dict[str, float] = {}
    for key in ["cpu_percent", "ram_percent", "ram_used_gb", "gpu_percent", "vram_used_gb"]:
        values = [sample.get(key, 0.0) for sample in samples]
        summary[f"{key}_start"] = round(first.get(key, 0.0), 1)
        summary[f"{key}_avg"] = round(sum(values) / len(values), 1)
        summary[f"{key}_peak"] = round(max(values), 1)
        summary[f"{key}_delta"] = round(max(values) - first.get(key, 0.0), 1)
    summary["ram_total_gb"] = round(samples[-1].get("ram_total_gb", 0.0), 1)
    summary["vram_total_gb"] = round(samples[-1].get("vram_total_gb", 0.0), 1)
    summary["sample_count"] = float(len(samples))
    summary["sample_window_seconds"] = float(max(0, len(samples) - 1))
    return summary


def _cpu_load_percent() -> float:
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "loadpercentage", "/value"],
            capture_output=True,
            text=True,
            timeout=3,
            encoding="utf-8",
            errors="replace",
            **_hidden_subprocess_kwargs(),
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("loadpercentage="):
                return float(line.split("=", 1)[1].strip())
    except Exception:
        return 0.0
    return 0.0


def _memory_status() -> tuple[float, float, float]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0.0, 0.0, 0.0
    total_gb = status.ullTotalPhys / (1024**3)
    used_gb = (status.ullTotalPhys - status.ullAvailPhys) / (1024**3)
    return float(status.dwMemoryLoad), used_gb, total_gb


def _nvidia_smi() -> tuple[float, float, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            encoding="utf-8",
            errors="replace",
            **_hidden_subprocess_kwargs(),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0, 0.0, 0.0
        first = result.stdout.strip().splitlines()[0]
        gpu_util, mem_used, mem_total = [float(part.strip()) for part in first.split(",")[:3]]
        return gpu_util, mem_used / 1024, mem_total / 1024
    except Exception:
        return 0.0, 0.0, 0.0


def _hidden_subprocess_kwargs() -> dict:
    kwargs: dict = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs
