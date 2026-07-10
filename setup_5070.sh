#!/usr/bin/env bash
# Environment setup for a local RTX 5070 (Blackwell, sm_120).
# The 5070 needs a CUDA 12.8+ PyTorch build; the default CPU/older-CUDA wheel
# will raise "no kernel image is available for execution on the device".
set -e

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Blackwell-capable PyTorch (CUDA 12.8). If the stable cu128 wheel is unavailable
# for your platform, use the nightly line below instead.
pip install torch --index-url https://download.pytorch.org/whl/cu128
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

pip install numpy scipy scikit-learn matplotlib pyyaml

echo "=== verifying GPU is visible and usable ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute capability: sm_%d%d" % cap)
    # tiny kernel test -- fails loudly if Blackwell kernels are missing
    x = torch.randn(256, 256, device="cuda"); y = (x @ x).sum()
    torch.cuda.synchronize()
    print("GPU matmul OK ->", float(y) is not None)
else:
    print("WARNING: no CUDA device; runs will fall back to CPU (much slower).")
PY
echo "=== setup complete. activate with: source .venv/bin/activate ==="
