from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from ctypes import wintypes

from .models import HardwareInfo


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class HardwareProbe:
    def probe(self) -> HardwareInfo:
        cpu_name = self._cpu_name()
        cpu_threads = os.cpu_count() or 1
        cpu_cores = max(1, cpu_threads // 2)
        ram_gb = self._ram_gb()
        gpu_name, vram_gb = self._gpu_info()
        drive, free_gb = self._preferred_drive()
        return HardwareInfo(
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            preferred_drive=drive,
            preferred_drive_free_gb=free_gb,
        )

    def _cpu_name(self) -> str:
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                timeout=5,
            ).strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
        return platform.processor() or platform.machine()

    def _ram_gb(self) -> float:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 1)
        return 0.0

    def _gpu_info(self) -> tuple[str, float]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            if out:
                name, memory_mb = [part.strip() for part in out.splitlines()[0].split(",", 1)]
                return name, round(float(memory_mb) / 1024, 1)
        except Exception:
            pass
        return "Unknown", 0.0

    def _preferred_drive(self) -> tuple[str, float]:
        candidates = ["E:\\", "D:\\", "C:\\"]
        best_drive = "C:\\"
        best_free = 0.0
        for drive in candidates:
            if not os.path.exists(drive):
                continue
            usage = shutil.disk_usage(drive)
            free = usage.free / (1024**3)
            if free > best_free:
                best_drive = drive
                best_free = free
        return best_drive, round(best_free, 1)
