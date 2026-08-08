# =============================================================================
# FASHN VTON - AWS EC2 GPU Deployment Script (PowerShell)
# =============================================================================
# Provisions a g4dn.xlarge EC2 instance with NVIDIA GPU, deploys the VTON
# server, and outputs connection details.
# =============================================================================

$ErrorActionPreference = "Stop"

# --- Configuration ---
$REGION        = "us-east-1"
$INSTANCE_TYPE = "g4dn.xlarge"
$KEY_NAME      = "fashn-vton-key"
$SG_NAME       = "fashn-vton-sg"
$KEY_FILE      = "fashn-vton-key.pem"
$VOLUME_SIZE   = 100  # GB - enough for weights + models + OS
$APP_DIR       = $PSScriptRoot  # Directory containing this script + app files

# --- AWS Credentials (from CSV) ---
$env:AWS_ACCESS_KEY_ID     = "YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
$env:AWS_DEFAULT_REGION    = $REGION

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " FASHN VTON - AWS EC2 GPU Deployment" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# ==========================================================================
# Step 1: Create / reuse Key Pair
# ==========================================================================
Write-Host "[1/5] Setting up SSH key pair..." -ForegroundColor Yellow

$existingKey = aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION 2>$null | ConvertFrom-Json
if ($existingKey.KeyPairs.Count -gt 0) {
    Write-Host "  Key pair '$KEY_NAME' already exists. Reusing." -ForegroundColor Green
} else {
    $keyResult = aws ec2 create-key-pair `
        --key-name $KEY_NAME `
        --query "KeyMaterial" `
        --output text `
        --region $REGION
    $keyResult | Out-File -FilePath (Join-Path $APP_DIR $KEY_FILE) -Encoding ASCII -NoNewline
    Write-Host "  Key pair created: $KEY_FILE" -ForegroundColor Green
    Write-Host "  IMPORTANT: Keep this file safe - it's your only way to SSH in!" -ForegroundColor Red
}

# ==========================================================================
# Step 2: Create / reuse Security Group
# ==========================================================================
Write-Host "[2/5] Setting up security group..." -ForegroundColor Yellow

# Get default VPC
$VPC_ID = (aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION).Trim()
Write-Host "  Default VPC: $VPC_ID"

$existingSg = aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --region $REGION 2>$null | ConvertFrom-Json
if ($existingSg.SecurityGroups.Count -gt 0) {
    $SG_ID = $existingSg.SecurityGroups[0].GroupId
    Write-Host "  Security group '$SG_NAME' already exists: $SG_ID" -ForegroundColor Green
} else {
    $SG_ID = (aws ec2 create-security-group `
        --group-name $SG_NAME `
        --description "FASHN VTON server - SSH + API access" `
        --vpc-id $VPC_ID `
        --query "GroupId" `
        --output text `
        --region $REGION).Trim()

    # SSH access (port 22)
    aws ec2 authorize-security-group-ingress `
        --group-id $SG_ID `
        --protocol tcp `
        --port 22 `
        --cidr "0.0.0.0/0" `
        --region $REGION | Out-Null

    # API access (port 8000)
    aws ec2 authorize-security-group-ingress `
        --group-id $SG_ID `
        --protocol tcp `
        --port 8000 `
        --cidr "0.0.0.0/0" `
        --region $REGION | Out-Null

    Write-Host "  Security group created: $SG_ID (SSH:22, API:8000 open)" -ForegroundColor Green
}

# ==========================================================================
# Step 3: Find the latest Deep Learning AMI (Ubuntu)
# ==========================================================================
Write-Host "[3/5] Finding NVIDIA Deep Learning AMI..." -ForegroundColor Yellow

$AMI_ID = (aws ec2 describe-images `
    --owners amazon `
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" `
    --query "sort_by(Images, &CreationDate)[-1].ImageId" `
    --output text `
    --region $REGION).Trim()

if ($AMI_ID -eq "None" -or [string]::IsNullOrEmpty($AMI_ID)) {
    # Fallback: try the standard Deep Learning AMI
    $AMI_ID = (aws ec2 describe-images `
        --owners amazon `
        --filters "Name=name,Values=Deep Learning AMI GPU PyTorch * (Ubuntu 22.04)*" "Name=state,Values=available" `
        --query "sort_by(Images, &CreationDate)[-1].ImageId" `
        --output text `
        --region $REGION).Trim()
}

if ($AMI_ID -eq "None" -or [string]::IsNullOrEmpty($AMI_ID)) {
    # Second fallback: any deep learning AMI with Ubuntu
    $AMI_ID = (aws ec2 describe-images `
        --owners amazon `
        --filters "Name=name,Values=*Deep Learning*Ubuntu*22.04*" "Name=state,Values=available" `
        --query "sort_by(Images, &CreationDate)[-1].ImageId" `
        --output text `
        --region $REGION).Trim()
}

Write-Host "  AMI: $AMI_ID" -ForegroundColor Green

# ==========================================================================
# Step 4: Build user-data script with embedded app code
# ==========================================================================
Write-Host "[4/5] Preparing user-data with embedded application code..." -ForegroundColor Yellow

# Read the app source files
$appPy        = [IO.File]::ReadAllText((Join-Path $APP_DIR "app.py"))
$enginePy     = [IO.File]::ReadAllText((Join-Path $APP_DIR "engine.py"))
$imageUtilsPy = [IO.File]::ReadAllText((Join-Path $APP_DIR "image_utils.py"))
$refinerPy    = [IO.File]::ReadAllText((Join-Path $APP_DIR "refiner.py"))
$requireTxt   = [IO.File]::ReadAllText((Join-Path $APP_DIR "requirements.txt"))

# Build the complete user-data script
$userData = @"
#!/bin/bash
set -euxo pipefail

APP_DIR="/opt/fashn-vton"
VENV_DIR="`${APP_DIR}/venv"
SERVICE_NAME="fashn-vton"

exec > /var/log/fashn-vton-setup.log 2>&1
echo "=== FASHN VTON setup started at `$(date) ==="

# Verify GPU
nvidia-smi || echo "WARNING: nvidia-smi failed, continuing anyway..."

# Create app directory
mkdir -p "`${APP_DIR}/weights"

# Write application files
cat > "`${APP_DIR}/app.py" << 'APPEOF'
${appPy}
APPEOF

cat > "`${APP_DIR}/engine.py" << 'ENGINEEOF'
${enginePy}
ENGINEEOF

cat > "`${APP_DIR}/image_utils.py" << 'IMGEOF'
${imageUtilsPy}
IMGEOF

cat > "`${APP_DIR}/refiner.py" << 'REFEOF'
${refinerPy}
REFEOF

cat > "`${APP_DIR}/requirements.txt" << 'REQEOF'
${requireTxt}
REQEOF

# System deps
apt-get update -y
apt-get install -y python3-venv python3-pip git

# Create venv
python3 -m venv "`${VENV_DIR}"
source "`${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# fashn_vton from GitHub (not on PyPI)
pip install "fashn_vton @ git+https://github.com/fashn-AI/fashn-vton.git"

# fashn_human_parser
pip install fashn-human-parser

# Remaining deps
pip install -r "`${APP_DIR}/requirements.txt"

# Systemd service
cat > /etc/systemd/system/`${SERVICE_NAME}.service << 'SVCEOF'
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
systemctl enable `${SERVICE_NAME}
systemctl start `${SERVICE_NAME}

echo "=== FASHN VTON setup complete at `$(date) ==="
"@

# Base64 encode the user-data
$userDataBytes  = [System.Text.Encoding]::UTF8.GetBytes($userData)
$userDataBase64 = [Convert]::ToBase64String($userDataBytes)

# Write to temp file for the CLI
$userDataFile = Join-Path $APP_DIR "userdata_temp.txt"
$userData | Out-File -FilePath $userDataFile -Encoding UTF8 -NoNewline

Write-Host "  User-data prepared (app code embedded)" -ForegroundColor Green

# ==========================================================================
# Step 5: Launch EC2 Instance
# ==========================================================================
Write-Host "[5/5] Launching EC2 instance ($INSTANCE_TYPE)..." -ForegroundColor Yellow

$INSTANCE_ID = (aws ec2 run-instances `
    --image-id $AMI_ID `
    --instance-type $INSTANCE_TYPE `
    --key-name $KEY_NAME `
    --security-group-ids $SG_ID `
    --block-device-mappings "[{`"DeviceName`":`"/dev/sda1`",`"Ebs`":{`"VolumeSize`":$VOLUME_SIZE,`"VolumeType`":`"gp3`"}}]" `
    --user-data "file://$userDataFile" `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fashn-vton-server}]" `
    --query "Instances[0].InstanceId" `
    --output text `
    --region $REGION).Trim()

Write-Host "  Instance launched: $INSTANCE_ID" -ForegroundColor Green
Write-Host "  Waiting for instance to be running..." -ForegroundColor Yellow

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
$PUBLIC_IP = (aws ec2 describe-instances `
    --instance-ids $INSTANCE_ID `
    --query "Reservations[0].Instances[0].PublicIpAddress" `
    --output text `
    --region $REGION).Trim()

# Clean up temp file
Remove-Item $userDataFile -Force -ErrorAction SilentlyContinue

# ==========================================================================
# Output
# ==========================================================================
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Instance ID : $INSTANCE_ID" -ForegroundColor White
Write-Host "  Public IP   : $PUBLIC_IP" -ForegroundColor White
Write-Host "  GPU         : NVIDIA T4 (16GB VRAM)" -ForegroundColor White
Write-Host ""
Write-Host "  SSH:" -ForegroundColor Cyan
Write-Host "    ssh -i $KEY_FILE ubuntu@$PUBLIC_IP" -ForegroundColor White
Write-Host ""
Write-Host "  API Endpoints:" -ForegroundColor Cyan
Write-Host "    Health:     http://${PUBLIC_IP}:8000/health" -ForegroundColor White
Write-Host "    Try-On:     http://${PUBLIC_IP}:8000/tryon" -ForegroundColor White
Write-Host "    Async API:  http://${PUBLIC_IP}:8000/api/try-on" -ForegroundColor White
Write-Host ""
Write-Host "  Monitor setup progress:" -ForegroundColor Cyan
Write-Host "    ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'tail -f /var/log/fashn-vton-setup.log'" -ForegroundColor White
Write-Host ""
Write-Host "  Check service status:" -ForegroundColor Cyan
Write-Host "    ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'sudo systemctl status fashn-vton'" -ForegroundColor White
Write-Host ""
Write-Host "  NOTE: First boot takes 10-15 minutes (installing PyTorch," -ForegroundColor Yellow
Write-Host "  downloading model weights, etc). The /health endpoint will" -ForegroundColor Yellow
Write-Host "  return 'model_loaded: true' once the server is fully ready." -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Green
