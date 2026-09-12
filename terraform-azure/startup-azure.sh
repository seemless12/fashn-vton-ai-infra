#!/bin/bash
set -ex

APP_DIR="/opt/fashn-vton"
VENV_DIR="/opt/fashn-vton/venv"
SERVICE_NAME="fashn-vton"

# Redirect all stdout & stderr to setup log for debugging
exec > /var/log/fashn-vton-setup.log 2>&1

echo "================================================================="
echo " Starting Shopping Buddy FASHN VTON Setup on Azure VM"
echo " Time: $(date)"
echo "================================================================="

# ---- 1. Wait for apt locks & install base system dependencies ----
export DEBIAN_FRONTEND=noninteractive
while fuser /var/lib/dpkg/lock >/dev/null 2>&1 || fuser /var/lib/apt/lists/lock >/dev/null 2>&1 ; do
    echo "Waiting for other package manager processes to complete..."
    sleep 3
done

apt-get update -y
apt-get install -y python3-venv python3-pip python3-dev git build-essential \
  curl wget ffmpeg libsm6 libxext6 libgl1-mesa-glx htop

# ---- 2. Ensure NVIDIA GPU Driver is ready ----
echo "Checking NVIDIA driver status..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Installing NVIDIA headless driver 535..."
    apt-get install -y linux-modules-nvidia-535-server-generic nvidia-headless-535-server nvidia-utils-535-server
fi

# Print GPU info if available
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || true
fi

# ---- 3. Create App Directory & Virtual Environment ----
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/weights"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip, wheel, setuptools
pip install --upgrade pip setuptools wheel

# ---- 4. Install PyTorch with CUDA 12.1 ----
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA in PyTorch
python3 -c "import torch; print(f'PyTorch CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}')" || true

# ---- 5. Install FASHN VTON and inference dependencies ----
echo "Installing fashn-vton package and dependencies..."
pip install "fashn_vton @ git+https://github.com/fashn-AI/fashn-vton-1.5.git"

pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
  "pillow>=10.0" "numpy>=1.24" "fashn-human-parser" \
  "onnxruntime-gpu>=1.17" "diffusers>=0.27" "accelerate>=0.28"

# ---- 6. Clone repo and download model weights ----
echo "Downloading FASHN model weights (~8GB)..."
cd "$APP_DIR"
if [ ! -d "repo" ]; then
    git clone https://github.com/fashn-AI/fashn-vton-1.5.git repo
fi

cd repo
source "$VENV_DIR/bin/activate"
python scripts/download_weights.py --weights-dir "$APP_DIR/weights" || true
cd "$APP_DIR"

# ---- 7. Setup Systemd Service ----
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Shopping Buddy FASHN VTON Inference Server on Azure
After=network.target

[Service]
User=root
WorkingDirectory=${APP_DIR}
Environment=PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=${VENV_DIR}/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

# Grant permissions to default azureuser so files can be transferred without root permissions
if id "azureuser" &>/dev/null; then
    chown -R azureuser:azureuser "$APP_DIR"
    chmod -R 775 "$APP_DIR"
fi

echo "================================================================="
echo " Setup complete! Waiting for application files to be uploaded."
echo " Once app.py is uploaded to $APP_DIR, run: sudo systemctl restart $SERVICE_NAME"
echo "================================================================="
