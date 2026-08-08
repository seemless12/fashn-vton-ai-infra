#!/bin/bash
set -euxo pipefail

APP_DIR="/opt/fashn-vton"
VENV_DIR="${APP_DIR}/venv"
SERVICE_NAME="fashn-vton"

exec > /var/log/fashn-vton-setup.log 2>&1
echo "=== FASHN VTON setup started at $(date) ==="

# Verify GPU
nvidia-smi || echo "WARNING: nvidia-smi failed, will retry after reboot..."

# Create app directory
mkdir -p "${APP_DIR}/weights"

# System deps
apt-get update -y
apt-get install -y python3-venv python3-pip python3-dev git build-essential

# Create venv
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# fashn_vton from GitHub (not on PyPI)
pip install "fashn_vton @ git+https://github.com/fashn-AI/fashn-vton.git"

# fashn_human_parser
pip install fashn-human-parser

# Write requirements.txt
cat > "${APP_DIR}/requirements.txt" << 'REQEOF'
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
pillow>=10.0
numpy>=1.24

onnxruntime>=1.17,<2.0

diffusers>=0.27
accelerate>=0.28
REQEOF

# Install remaining deps
pip install -r "${APP_DIR}/requirements.txt"

echo "=== Python packages installed at $(date) ==="

# Systemd service
cat > /etc/systemd/system/${SERVICE_NAME}.service << 'SVCEOF'
[Unit]
Description=FASHN VTON AI Try-On Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/fashn-vton
Environment="PATH=/opt/fashn-vton/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin"
Environment="FASHN_WEIGHTS_DIR=/opt/fashn-vton/weights"
Environment="HF_HOME=/opt/fashn-vton/.cache/huggingface"
ExecStart=/opt/fashn-vton/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

echo "=== Dependencies installed, service configured. App files will be uploaded via SCP. ==="
echo "=== FASHN VTON bootstrap complete at $(date) ==="
