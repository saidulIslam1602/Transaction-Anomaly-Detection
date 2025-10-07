output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "aks_cluster_name" {
  description = "Name of the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.name
}

output "aks_cluster_id" {
  description = "ID of the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.id
}

output "acr_login_server" {
  description = "Login server for Azure Container Registry"
  value       = azurerm_container_registry.main.login_server
}

output "acr_admin_username" {
  description = "Admin username for ACR"
  value       = azurerm_container_registry.main.admin_username
  sensitive   = true
}

output "acr_admin_password" {
  description = "Admin password for ACR"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.main.name
}

output "storage_account_primary_connection_string" {
  description = "Primary connection string for storage account"
  value       = azurerm_storage_account.main.primary_connection_string
  sensitive   = true
}

output "key_vault_id" {
  description = "ID of the Key Vault"
  value       = azurerm_key_vault.main.id
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "databricks_workspace_url" {
  description = "URL of the Databricks workspace"
  value       = azurerm_databricks_workspace.main.workspace_url
}

output "databricks_workspace_id" {
  description = "ID of the Databricks workspace"
  value       = azurerm_databricks_workspace.main.id
}

output "ml_workspace_id" {
  description = "ID of the Machine Learning workspace"
  value       = azurerm_machine_learning_workspace.main.id
}

output "application_insights_instrumentation_key" {
  description = "Instrumentation key for Application Insights"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}

output "redis_hostname" {
  description = "Hostname of Redis cache"
  value       = azurerm_redis_cache.main.hostname
}

output "redis_primary_access_key" {
  description = "Primary access key for Redis cache"
  value       = azurerm_redis_cache.main.primary_access_key
  sensitive   = true
}

output "event_hub_namespace_name" {
  description = "Name of the Event Hub namespace"
  value       = azurerm_eventhub_namespace.main.name
}

output "kube_config" {
  description = "Kubernetes configuration for AKS"
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

