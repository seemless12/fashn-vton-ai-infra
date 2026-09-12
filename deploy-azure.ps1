# =============================================================================
# Shopping Buddy AI - Microsoft Azure Deployment & Management Script
# =============================================================================
# Manages the Azure GPU VM lifecycle:
#   - Provisioning via Terraform
#   - Syncing code & model files to /opt/fashn-vton
#   - Restarting & checking the FastAPI inference server
#   - Stopping/Starting the VM for zero-cost cold idle
# =============================================================================

param (
    [Parameter(Position = 0)]
    [ValidateSet("provision", "deploy-code", "start", "stop", "status", "health", "destroy")]
    [string]$Action = "status",

    [string]$Region = "centralindia",
    [string]$VmSize = "Standard_NC4as_T4_v3"
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$TerraformDir = Join-Path $ScriptDir "terraform-azure"
$TerraformExe = Join-Path $ScriptDir "terraform\terraform.exe"
$KeyPath = Join-Path $TerraformDir "fashn-vton-azure-key.pem"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Shopping Buddy - Microsoft Azure AI Infrastructure Manager" -ForegroundColor Cyan
Write-Host " Action: $Action" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

function Get-AzureOutputs {
    if (-not (Test-Path $TerraformExe)) {
        throw "Terraform executable not found at $TerraformExe"
    }
    Push-Location $TerraformDir
    try {
        $ip = & $TerraformExe output -raw public_ip_address 2>$null
        $api = & $TerraformExe output -raw api_endpoint 2>$null
        $rg = & $TerraformExe output -raw resource_group 2>$null
        return @{
            PublicIp = $ip
            ApiEndpoint = $api
            ResourceGroup = $rg
        }
    } finally {
        Pop-Location
    }
}

switch ($Action) {
    "provision" {
        Write-Host "`n[1/3] Initializing Terraform for Azure..." -ForegroundColor Yellow
        Push-Location $TerraformDir
        try {
            & $TerraformExe init
            Write-Host "`n[2/3] Planning and applying infrastructure..." -ForegroundColor Yellow
            & $TerraformExe apply -auto-approve -var="azure_region=$Region" -var="vm_size=$VmSize"
            Write-Host "`n[3/3] Infrastructure provisioned successfully!" -ForegroundColor Green
            & $TerraformExe output
        } finally {
            Pop-Location
        }
    }

    "deploy-code" {
        $info = Get-AzureOutputs
        $ip = $info.PublicIp
        if ([string]::IsNullOrWhiteSpace($ip)) {
            Write-Host "Error: No public IP found. Run './deploy-azure.ps1 provision' first." -ForegroundColor Red
            return
        }

        Write-Host "`nTarget Server: $ip" -ForegroundColor Yellow
        Write-Host "Syncing application code to /opt/fashn-vton..." -ForegroundColor Yellow

        $filesToUpload = @(
            "app.py",
            "license_manager.py",
            "engine_optimized.py",
            "refiner.py",
            "image_utils.py",
            "requirements.txt"
        )

        foreach ($file in $filesToUpload) {
            $localFilePath = Join-Path $ScriptDir $file
            if (Test-Path $localFilePath) {
                Write-Host "  Uploading $file..." -ForegroundColor Gray
                scp -o StrictHostKeyChecking=no -i $KeyPath $localFilePath "azureuser@${ip}:/opt/fashn-vton/$file"
            }
        }

        Write-Host "`nRestarting fashn-vton systemd service..." -ForegroundColor Yellow
        ssh -o StrictHostKeyChecking=no -i $KeyPath "azureuser@${ip}" "sudo systemctl restart fashn-vton"

        Write-Host "`nChecking service status..." -ForegroundColor Yellow
        ssh -o StrictHostKeyChecking=no -i $KeyPath "azureuser@${ip}" "sudo systemctl status fashn-vton --no-pager"

        Write-Host "`nCode deployment complete!" -ForegroundColor Green
    }

    "start" {
        Write-Host "`nStarting / Allocating Azure GPU VM..." -ForegroundColor Yellow
        $info = Get-AzureOutputs
        if ($info.ResourceGroup) {
            # Uses Azure CLI if installed
            az vm start --resource-group $info.ResourceGroup --name "fashn-vton-vm" --no-wait
            Write-Host "VM start command sent. GPU server will be available in ~30-45 seconds." -ForegroundColor Green
        } else {
            Write-Host "Could not find Resource Group. Please start from Azure Portal." -ForegroundColor Red
        }
    }

    "stop" {
        Write-Host "`nDeallocating Azure GPU VM to stop billing ($0.00/hr)..." -ForegroundColor Yellow
        $info = Get-AzureOutputs
        if ($info.ResourceGroup) {
            az vm deallocate --resource-group $info.ResourceGroup --name "fashn-vton-vm"
            Write-Host "VM deallocated successfully! Compute costs are now $0.00/hr." -ForegroundColor Green
        } else {
            Write-Host "Please stop and deallocate 'fashn-vton-vm' via the Azure Portal." -ForegroundColor Red
        }
    }

    "status" {
        $info = Get-AzureOutputs
        Write-Host "Public IP:       $($info.PublicIp)" -ForegroundColor White
        Write-Host "API Endpoint:    $($info.ApiEndpoint)" -ForegroundColor White
        Write-Host "Resource Group:  $($info.ResourceGroup)" -ForegroundColor White
        Write-Host "SSH Key File:    $KeyPath" -ForegroundColor White
        Write-Host "`nSSH Command:" -ForegroundColor Cyan
        Write-Host "ssh -i `"$KeyPath`" azureuser@$($info.PublicIp)" -ForegroundColor Yellow
    }

    "health" {
        $info = Get-AzureOutputs
        $ip = $info.PublicIp
        if ([string]::IsNullOrWhiteSpace($ip)) {
            Write-Host "No Public IP available." -ForegroundColor Red
            return
        }
        $url = "http://${ip}:8000/health"
        Write-Host "Querying health check at $url..." -ForegroundColor Yellow
        try {
            $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 10
            Write-Host "Health Check Status: OK" -ForegroundColor Green
            $resp | Format-List
        } catch {
            Write-Host "Health check failed or server is still starting up: $_" -ForegroundColor Red
        }
    }

    "destroy" {
        $confirm = Read-Host "Are you sure you want to completely DESTROY all Azure resources? (yes/no)"
        if ($confirm -eq "yes") {
            Push-Location $TerraformDir
            try {
                & $TerraformExe destroy -auto-approve
                Write-Host "All Azure resources successfully destroyed." -ForegroundColor Green
            } finally {
                Pop-Location
            }
        } else {
            Write-Host "Destroy aborted." -ForegroundColor Gray
        }
    }
}
