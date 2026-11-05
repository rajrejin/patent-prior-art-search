"""Detect hardware and library versions, write results/environment.json."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED = [
    "pydantic",
    "typer",
    "httpx",
    "lxml",
    "numpy",
    "torch",
    "sentence-transformers",
    "chromadb",
    "rank-bm25",
    "neo4j",
    "fastapi",
    "mcp",
    "streamlit",
]


def _total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [  # noqa: RUF012
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1024**3, 2)
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 1024**3, 2)
    except Exception:
        return None


def _tool_version(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    libs: dict[str, str | None] = {}
    for name in TRACKED:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "python": sys.version.split()[0],
        "gpu": "none assumed (CPU-only target)",
        "docker": _tool_version(["docker", "--version"]),
        "ollama": _tool_version(["ollama", "--version"]),
        "git": _tool_version(["git", "--version"]),
        "libraries": libs,
    }


def write_environment(path: Path) -> dict[str, Any]:
    env = collect_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env
