variable "azure_region" {
  description = "Azure region for resources. centralindia (Pune) has low latency to PK/IN, or use eastus"
  type        = string
  default     = "centralindia"
}

variable "project_name" {
  description = "Prefix for all Azure resource names"
  type        = string
  default     = "fashn-vton"
}

variable "vm_size" {
  description = "Azure VM size. Standard_NC4as_T4_v3 (NVIDIA T4 16GB, $0.526/hr) or Standard_NV36ads_A10_v5 (A10 24GB, $1.98/hr)"
  type        = string
  default     = "Standard_NC4as_T4_v3"
}

variable "admin_username" {
  description = "Admin username for the Azure Linux VM"
  type        = string
  default     = "azureuser"
}

variable "key_name" {
  description = "File name for the generated SSH private key (.pem)"
  type        = string
  default     = "fashn-vton-azure-key"
}

variable "disk_size_gb" {
  description = "OS Disk size in GB (needs ~100GB for OS + PyTorch + FASHN weights)"
  type        = number
  default     = 100
}

variable "app_port" {
  description = "Port where FastAPI server listens"
  type        = number
  default     = 8000
}
