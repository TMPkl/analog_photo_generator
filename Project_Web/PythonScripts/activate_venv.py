import os
import sys
import subprocess
from pathlib import Path
import shutil

def run(cmd, cwd=None, retries=1):
    for attempt in range(1, retries + 1):
        print(f"\n[Attempt {attempt}/{retries}] Running: {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=cwd)
        if result.returncode == 0:
            return True
        print(f"⚠️ Command failed (code {result.returncode}). Retrying...\n")
    print(f"❌ Command failed after {retries} attempts: {cmd}")
    sys.exit(result.returncode)

def find_python310():
    candidates = ["python3.10", "/usr/bin/python3.10", "C:\\Python310\\python.exe"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    print("❌ Python 3.10 not found on this system.")
    print("Please install Python 3.10 and ensure it's available as 'python3.10'.")
    sys.exit(1)

# 🔎 Ensure we find python3.10 first
python310 = find_python310()
print(f"✅ Found Python 3.10 at: {python310}")

# 1. Create .venv using python3.10
venv_dir = Path(".venv").resolve()
if not venv_dir.exists():
    print("📦 Creating virtual environment with Python 3.10...")
    run(f"{python310} -m venv .venv")
else:
    print("✅ .venv already exists.")

# 2. Use the venv's Python from now on
venv_python = venv_dir / ("Scripts/python.exe" if os.name == 'nt' else "bin/python")
venv_python = str(venv_python)

# 3. Confirm venv Python is correct version
result = subprocess.run(f"{venv_python} --version", shell=True, capture_output=True, text=True)
print(f"📦 Using venv Python: {result.stdout.strip()}")

# 4. Upgrade pip
run(f"{venv_python} -m pip install --upgrade pip", retries=3)

# 5. Install main packages (excluding TensorFlow)
packages = [
    "pyyaml==5.1",
    "onnxruntime",
    "rembg",
    "opencv-python",
    "mtcnn",
    "tf-keras",
    "retina-face",
    "easyocr"
]
run(f"{venv_python} -m pip install --timeout 300 --retries 10 --prefer-binary " + " ".join(packages))

print("\n✅ Environment setup complete. Python 3.10 venv is ready.")
