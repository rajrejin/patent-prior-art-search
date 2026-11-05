# Create the venv and install everything (CPU-only PyTorch).
$ErrorActionPreference = "Stop"
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[dev,ml,serve]"
