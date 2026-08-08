#!/bin/bash
set -ex

APP_DIR="/opt/fashn-vton"
VENV_DIR="/opt/fashn-vton/venv"
SERVICE_NAME="fashn-vton"

exec > /var/log/fashn-vton-setup.log 2>&1

# ---- System packages ----
apt-get update -y
apt-get install -y python3-venv python3-pip git

# ---- App directory & venv ----
mkdir -p "$APP_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# ---- Install PyTorch with CUDA ----
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---- Install fashn_vton from GitHub ----
pip install "fashn_vton @ git+https://github.com/fashn-AI/fashn-vton-1.5.git"

# ---- Install remaining requirements ----
pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
  "pillow>=10.0" "numpy>=1.24" "fashn-human-parser" \
  "onnxruntime>=1.17,<2.0" "diffusers>=0.27" "accelerate>=0.28"

# ---- Download model weights ----
cd "$APP_DIR"
git clone https://github.com/fashn-AI/fashn-vton-1.5.git repo
source "$VENV_DIR/bin/activate"
cd repo
python scripts/download_weights.py --weights-dir "$APP_DIR/weights"
cd "$APP_DIR"

# ---- Systemd service ----
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=FASHN VTON Inference Server
After=network.target

[Service]
User=root
WorkingDirectory=${APP_DIR}
Environment=PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=${VENV_DIR}/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

# NOTE: The service will fail to start until app.py is uploaded.
# After deploying, SCP your app files and run:
#   sudo systemctl start fashn-vton
