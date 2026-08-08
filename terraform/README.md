# FASHN VTON — Terraform Deployment

Deploy the FASHN VTON inference server to **any AWS account** with a single command.

## Prerequisites

1. [Terraform](https://developer.hashicorp.com/terraform/install) installed
2. AWS CLI configured with credentials (`aws configure`)
3. GPU quota: at least **4 vCPUs** for G-type instances in your target region

## Quick Start

```powershell
cd terraform

# 1. Initialize Terraform
terraform init

# 2. Preview what will be created
terraform plan

# 3. Deploy everything
terraform apply -auto-approve

# 4. Upload your app code (use the SSH key Terraform generated)
scp -i fashn-vton-key.pem app.py engine.py image_utils.py refiner.py ubuntu@<PUBLIC_IP>:/opt/fashn-vton/

# 5. Start the service
ssh -i fashn-vton-key.pem ubuntu@<PUBLIC_IP> "sudo systemctl start fashn-vton"
```

## Deploy to a Different Region or Account

```powershell
# Different region
terraform apply -var="aws_region=us-east-1"

# Different instance type
terraform apply -var="instance_type=g6.xlarge"

# Use a different AWS account — just change your credentials
$env:AWS_ACCESS_KEY_ID="NEW_KEY"
$env:AWS_SECRET_ACCESS_KEY="NEW_SECRET"
terraform apply
```

## Tear Down (stop all costs)

```powershell
terraform destroy -auto-approve
```

## Files

| File | Purpose |
|------|---------|
| `main.tf` | Core infrastructure (EC2, SG, Key Pair) |
| `variables.tf` | Configurable parameters |
| `outputs.tf` | Deployment info (IP, SSH command, etc.) |
| `startup.sh` | EC2 bootstrap (installs deps, downloads weights) |
