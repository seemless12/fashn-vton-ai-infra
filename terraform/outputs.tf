output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.fashn_vton.id
}

output "public_ip" {
  description = "Public IP of the FASHN VTON server"
  value       = aws_instance.fashn_vton.public_ip
}

output "api_endpoint" {
  description = "Full API endpoint URL"
  value       = "http://${aws_instance.fashn_vton.public_ip}:${var.app_port}"
}

output "health_check_url" {
  description = "Health check URL"
  value       = "http://${aws_instance.fashn_vton.public_ip}:${var.app_port}/health"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${local_file.private_key.filename} ubuntu@${aws_instance.fashn_vton.public_ip}"
}

output "ssh_private_key_path" {
  description = "Path to the SSH private key file"
  value       = local_file.private_key.filename
}

output "ami_id" {
  description = "AMI ID used"
  value       = data.aws_ami.deep_learning.id
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "stop_command" {
  description = "Command to stop the instance (save money)"
  value       = "aws ec2 stop-instances --instance-ids ${aws_instance.fashn_vton.id} --region ${var.aws_region}"
}

output "start_command" {
  description = "Command to start the instance again"
  value       = "aws ec2 start-instances --instance-ids ${aws_instance.fashn_vton.id} --region ${var.aws_region}"
}
