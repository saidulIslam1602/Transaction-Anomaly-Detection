variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "norwayeast"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.28.0"
}

variable "aks_node_count" {
  description = "Initial number of nodes in AKS cluster"
  type        = number
  default     = 3
}

variable "allowed_ip_ranges" {
  description = "Allowed IP ranges for Key Vault access"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "enable_databricks" {
  description = "Enable Azure Databricks workspace"
  type        = bool
  default     = true
}

variable "enable_ml_workspace" {
  description = "Enable Azure Machine Learning workspace"
  type        = bool
  default     = true
}

variable "enable_event_hub" {
  description = "Enable Event Hub for streaming"
  type        = bool
  default     = true
}

variable "redis_cache_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 2
}

variable "storage_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "GRS"

  validation {
    condition     = contains(["LRS", "GRS", "RAGRS", "ZRS"], var.storage_replication_type)
    error_message = "Storage replication must be LRS, GRS, RAGRS, or ZRS."
  }
}

