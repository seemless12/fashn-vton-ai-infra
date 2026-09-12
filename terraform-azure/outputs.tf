output "public_ip_address" {
  description = "Public IP address of the Azure GPU VM"
  value       = azurerm_public_ip.public_ip.ip_address
}

output "api_endpoint" {
  description = "Public API endpoint for FASHN VTON Inference"
  value       = "http://${azurerm_public_ip.public_ip.ip_address}:${var.app_port}"
}

output "health_check_url" {
  description = "Health check URL"
  value       = "http://${azurerm_public_ip.public_ip.ip_address}:${var.app_port}/health"
}

output "ssh_command" {
  description = "Command to SSH into the Azure GPU VM"
  value       = "ssh -i ${var.key_name}.pem ${var.admin_username}@${azurerm_public_ip.public_ip.ip_address}"
}

output "vm_size" {
  description = "Azure VM compute size"
  value       = var.vm_size
}

output "resource_group" {
  description = "Azure Resource Group name"
  value       = azurerm_resource_group.rg.name
}
