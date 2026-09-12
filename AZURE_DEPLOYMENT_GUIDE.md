# Shopping Buddy AI - Microsoft Azure Deployment Guide

This guide provides step-by-step instructions for deploying the Shopping Buddy FASHN VTON AI diffusion backend to **Microsoft Azure**.

---

## Why Microsoft Azure?

1. **Card Acceptance (Pakistan & International)**: Unlike GCP which frequently declines Pakistani debit cards, Azure natively supports **SadaPay Mastercard** and **NayaPay Visa** with 3D Secure OTP verification.
2. **\$200 Free Trial**: Microsoft provides \$200 (approx. PKR 56,000) in free credits valid for 30 days.
3. **Superior Economics**: The recommended `Standard_NC4as_T4_v3` instance costs **\$0.526/hr**, which is **47% cheaper** than AWS `g5.xlarge` (\$1.006/hr).
4. **Profit Margins**: At PKR 0.44 per generation on Azure, the PKR 500 subscription delivers **over 91% gross margin**.

| Platform | VM Instance | GPU Hardware | Hourly Rate | Est. Time / Image | Compute Cost / Image |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AWS** | `g5.xlarge` | NVIDIA A10G (24GB) | \$1.006 / hr | ~7.3s | PKR 0.66 |
| **Azure (Recommended)** | `Standard_NC4as_T4_v3` | NVIDIA Tesla T4 (16GB) | **\$0.526 / hr** | ~9.5s | **PKR 0.44** |
| **Azure (High-Performance)** | `Standard_NV36ads_A10_v5` | NVIDIA A10 (24GB) | \$1.980 / hr | ~7.1s | PKR 1.25 |

---

## Step 1: Create Azure Account & Claim \$200 Free Credits

1. Go to [azure.microsoft.com/free](https://azure.microsoft.com/free).
2. Sign in with your Microsoft account (Outlook / Hotmail / Gmail).
3. Under Payment Information, enter your **SadaPay Mastercard** or **NayaPay Visa**.
   - Make sure **International Transactions** and **Online Payments** are enabled in your SadaPay / NayaPay app.
   - Azure will perform a temporary verification charge (around \$1, instantly refunded) with a 3D Secure OTP sent to your phone.
4. Once verified, your subscription will display **\$200.00 credit available**.

---

## Step 2: Request GPU Quota Increase (Mandatory)

> [!IMPORTANT]
> All new Azure subscriptions have a default quota of **0 vCPUs** for GPU instances to prevent accidental charges. You must request a quota increase before provisioning the GPU VM. Approvals are typically automated and take 5–15 minutes.

1. In the [Azure Portal](https://portal.azure.com), search for **Quotas** in the top search bar.
2. Click **Compute**.
3. Set the region to **Central India** (Pune - lowest latency to Pakistan) or **East US**.
4. In the search box, type: `NCas_T4_v3` (or `Standard NCas_T4_v3 Family vCPUs`).
5. Click the pencil icon / **Request Increase**:
   - Enter new limit: **4** (or **8**).
   - Submit request.
6. Refresh after 5–10 minutes until the status shows **Approved**.

---

## Step 3: Deployment Options

You have three ways to deploy:
- **Option A (Fastest / Zero Setup)**: Azure Cloud Shell directly in your browser.
- **Option B (PowerShell / Local Terraform)**: Automated scripts from your Windows PC.
- **Option C (Azure Portal GUI)**: 1-Click web console setup.

---

### Option A: Azure Cloud Shell (Recommended)

Azure Cloud Shell runs in your browser with Terraform and Azure CLI pre-installed.

1. In the [Azure Portal](https://portal.azure.com), click the **Cloud Shell** icon (`>_`) in the top navigation bar.
2. Select **Bash**. (If prompted, create default storage).
3. Clone your repository:
   ```bash
   git clone https://github.com/<YOUR_GITHUB>/fashn-vton-ai-infra.git
   cd fashn-vton-ai-infra/terraform-azure
   ```
4. Initialize and launch the GPU VM:
   ```bash
   terraform init
   terraform apply -auto-approve
   ```
5. Note the outputs:
   - `public_ip_address`: Your server's public IP.
   - `api_endpoint`: `http://<IP>:8000`
   - `ssh_command`: Ready-to-use SSH connection command.

---

### Option B: Local Deployment with PowerShell

If you have Azure CLI (`az`) installed locally:

1. Open PowerShell and navigate to `fashn-vton-ai-infra`:
   ```powershell
   az login
   ```
2. Run the deployment manager:
   ```powershell
   .\deploy-azure.ps1 provision
   ```
3. Upload the app code and restart the server:
   ```powershell
   .\deploy-azure.ps1 deploy-code
   ```
4. Verify the server health:
   ```powershell
   .\deploy-azure.ps1 health
   ```

---

### Option C: Azure Portal GUI (Manual Creation)

If you prefer clicking through the Azure Portal UI:

1. Navigate to **Virtual Machines** $\rightarrow$ **Create** $\rightarrow$ **Azure virtual machine**.
2. **Basics Tab**:
   - **Resource Group**: Create new `fashn-vton-rg`.
   - **Virtual Machine Name**: `fashn-vton-vm`.
   - **Region**: `Central India` (or `East US`).
   - **Image**: `Ubuntu Server 22.04 LTS - x64 Gen2`.
   - **Size**: Select `Standard_NC4as_T4_v3`.
   - **Authentication Type**: SSH Public Key (Generate new key pair or upload existing).
3. **Disks Tab**:
   - **OS Disk Size**: Change to **100 GiB** (or larger).
   - **OS Disk Type**: Premium SSD.
4. **Networking Tab**:
   - Allow ports: **SSH (22)**.
   - Under Network Security Group, click **Add inbound rule**:
     - Destination port: `8000`
     - Protocol: `TCP`
     - Name: `FastAPI`
5. **Advanced Tab**:
   - Scroll down to **User Data** (Cloud-init).
   - Copy and paste the entire contents of [`terraform-azure/startup-azure.sh`](terraform-azure/startup-azure.sh).
6. Click **Review + Create** $\rightarrow$ **Create**.

---

## Step 4: Upload Application Code & Start Inference

Once the VM is running and setup completes (~10-12 minutes for CUDA + weights download):

1. Copy the application files to the VM:
   ```bash
   scp -i terraform-azure/fashn-vton-azure-key.pem \
       app.py license_manager.py engine_optimized.py refiner.py image_utils.py requirements.txt \
       azureuser@<AZURE_PUBLIC_IP>:/opt/fashn-vton/
   ```

2. SSH into the VM:
   ```bash
   ssh -i terraform-azure/fashn-vton-azure-key.pem azureuser@<AZURE_PUBLIC_IP>
   ```

3. Restart and verify the service:
   ```bash
   sudo systemctl restart fashn-vton
   sudo systemctl status fashn-vton
   ```

4. View setup and runtime logs:
   ```bash
   # Setup log:
   tail -f /var/log/fashn-vton-setup.log

   # Application server log:
   journalctl -u fashn-vton -f
   ```

5. Test the health endpoint:
   ```bash
   curl http://<AZURE_PUBLIC_IP>:8000/health
   ```
   Output:
   ```json
   {
     "status": "healthy",
     "gpu_available": true,
     "device": "cuda:0",
     "device_name": "Tesla T4"
   }
   ```

---

## Step 5: Update Chrome Extension & Web App

Update the API base URL in your frontend and extension to point to your Azure server:

### Chrome Extension (`fashn-extension`)
In `fashn-extension/content.js` and `fashn-extension/popup.js`:
```javascript
const FASHN_API_URL = "http://<AZURE_PUBLIC_IP>:8000";
```

### React Web Frontend (`frontend`)
In `frontend/.env.production`:
```env
VITE_API_BASE_URL=http://<AZURE_PUBLIC_IP>:8000
```

---

## Step 6: Cost Management & Cold-Idle Mode (\$0.00/hr)

When you are not doing demonstrations or running inferences, **deallocate** the virtual machine so Azure charges **\$0.00/hr** for compute:

### Via Azure Portal:
1. Open **Virtual Machines** $\rightarrow$ `fashn-vton-vm`.
2. Click **Stop**. Ensure status transitions to **Stopped (deallocated)**.
3. When ready to present, click **Start**. The VM Boots with all weights preserved in ~45 seconds.

### Via PowerShell Script:
```powershell
# Stop and stop billing:
.\deploy-azure.ps1 stop

# Start back up for presentation:
.\deploy-azure.ps1 start
```

### Via Azure CLI:
```bash
# Stop:
az vm deallocate --resource-group fashn-vton-rg --name fashn-vton-vm

# Start:
az vm start --resource-group fashn-vton-rg --name fashn-vton-vm
```
