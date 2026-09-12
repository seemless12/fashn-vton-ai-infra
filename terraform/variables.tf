variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "ap-south-1" # Mumbai
}

variable "instance_type" {
  description = "EC2 instance type (must have GPU)"
  type        = string
  default     = "g5.xlarge" # NVIDIA A10G, 24GB VRAM
}

variable "ebs_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 100
}

variable "key_name" {
  description = "Name for the SSH key pair"
  type        = string
  default     = "fashn-vton-key-v2"
}

variable "app_port" {
  description = "Port the FASHN VTON API listens on"
  type        = number
  default     = 8000
}

variable "project_name" {
  description = "Project name used for tagging resources"
  type        = string
  default     = "fashn-vton-v2"
}
