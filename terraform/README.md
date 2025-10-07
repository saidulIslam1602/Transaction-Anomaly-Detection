# Terraform Infrastructure for Transaction Anomaly Detection

This directory contains Terraform configuration for deploying the complete Azure infrastructure for the Transaction Anomaly Detection system.

## Prerequisites

1. **Azure CLI** installed and configured
2. **Terraform** >= 1.5.0 installed
3. **Azure subscription** with appropriate permissions
4. **Service Principal** with Contributor access

## Infrastructure Components

The Terraform configuration creates the following Azure resources:

### Core Infrastructure
- **Resource Group**: Container for all resources
- **Virtual Network**: Network infrastructure with subnets
- **Network Security Groups**: Security rules for network traffic

### Compute & Containers
- **Azure Kubernetes Service (AKS)**: Managed Kubernetes cluster
  - Auto-scaling node pools
  - Azure CNI networking
  - Azure Monitor integration
- **Azure Container Registry (ACR)**: Docker image registry
  - Premium SKU with geo-replication
  - Admin access enabled

### Storage
- **Storage Account**: Blob storage for data and models
  - Containers: ml-data, ml-models, feature-store
  - Versioning and soft delete enabled
- **Redis Cache**: In-memory caching for features
  - Standard tier, 2 capacity

### Data & ML Services
- **Azure Databricks**: Data engineering and ML platform
  - Premium tier with VNet injection
  - Separate public/private subnets
- **Azure Machine Learning**: ML workspace
  - Integrated with Key Vault and Application Insights
- **Event Hub**: Streaming data ingestion
  - For real-time transaction processing

### Security & Monitoring
- **Key Vault**: Secrets and key management
  - Premium tier with soft delete
  - Network ACLs enabled
- **Log Analytics Workspace**: Centralized logging
- **Application Insights**: Application monitoring

## Quick Start

### 1. Initialize Terraform

```bash
cd terraform
terraform init
```

### 2. Configure Variables

Copy the example variables file and update with your values:

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your configuration
```

### 3. Review the Plan

```bash
terraform plan -out=tfplan
```

### 4. Apply the Configuration

```bash
terraform apply tfplan
```

### 5. Retrieve Outputs

```bash
# Get all outputs
terraform output

# Get specific output
terraform output aks_cluster_name

# Get sensitive outputs
terraform output -raw kube_config > ~/.kube/config
```

## Configuration Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `environment` | Environment name (dev/staging/prod) | `prod` | No |
| `location` | Azure region | `norwayeast` | No |
| `kubernetes_version` | AKS Kubernetes version | `1.28.0` | No |
| `aks_node_count` | Initial AKS node count | `3` | No |
| `allowed_ip_ranges` | IP ranges for Key Vault access | `[]` | Yes |
| `enable_databricks` | Enable Databricks workspace | `true` | No |
| `enable_ml_workspace` | Enable ML workspace | `true` | No |
| `enable_event_hub` | Enable Event Hub | `true` | No |

## Important Outputs

After applying, Terraform will output important values:

- `aks_cluster_name`: AKS cluster name for kubectl configuration
- `acr_login_server`: Docker registry URL
- `databricks_workspace_url`: Databricks workspace URL
- `key_vault_uri`: Key Vault URI for secrets

## State Management

The Terraform state is stored in Azure Blob Storage for team collaboration:

```hcl
backend "azurerm" {
  resource_group_name  = "rg-aml-terraform-state"
  storage_account_name = "amltfstate"
  container_name       = "tfstate"
  key                  = "prod.terraform.tfstate"
}
```

### Setting Up State Storage

Before first run, create the state storage:

```bash
# Create resource group
az group create --name rg-aml-terraform-state --location norwayeast

# Create storage account
az storage account create \
  --name amltfstate \
  --resource-group rg-aml-terraform-state \
  --location norwayeast \
  --sku Standard_LRS

# Create container
az storage container create \
  --name tfstate \
  --account-name amltfstate
```

## Post-Deployment Steps

### 1. Configure kubectl

```bash
az aks get-credentials \
  --resource-group $(terraform output -raw resource_group_name) \
  --name $(terraform output -raw aks_cluster_name)
```

### 2. Store Secrets in Key Vault

```bash
# Store OpenAI API key
az keyvault secret set \
  --vault-name $(terraform output -raw key_vault_name) \
  --name openai-api-key \
  --value "your-openai-key"

# Store storage connection string
az keyvault secret set \
  --vault-name $(terraform output -raw key_vault_name) \
  --name azure-storage-connection-string \
  --value "$(terraform output -raw storage_account_primary_connection_string)"
```

### 3. Push Docker Image to ACR

```bash
# Login to ACR
az acr login --name $(terraform output -raw acr_login_server)

# Tag and push image
docker tag transaction-anomaly-detection:latest \
  $(terraform output -raw acr_login_server)/transaction-anomaly-detection:latest

docker push $(terraform output -raw acr_login_server)/transaction-anomaly-detection:latest
```

## Cost Estimation

Estimated monthly costs (Norway East region):

| Service | Configuration | Est. Cost (USD) |
|---------|--------------|-----------------|
| AKS | 3x Standard_D4s_v3 | ~$400 |
| ACR | Premium with geo-replication | ~$40 |
| Storage | 500GB with GRS | ~$50 |
| Databricks | Premium tier | ~$200 |
| Redis Cache | Standard, 2 capacity | ~$75 |
| Event Hub | Standard, 2 throughput units | ~$45 |
| Key Vault | Premium | ~$10 |
| **Total** | | **~$820/month** |

## Security Considerations

1. **Network Security**:
   - VNet injection for Databricks
   - Private endpoints for storage
   - Network policies in AKS

2. **Identity & Access**:
   - Managed identities for services
   - RBAC for AKS and ACR
   - Key Vault for secrets

3. **Compliance**:
   - Soft delete enabled on storage
   - Audit logging enabled
   - Data encryption at rest

## Maintenance

### Update Infrastructure

```bash
# Pull latest Terraform changes
git pull

# Plan and apply updates
terraform plan -out=tfplan
terraform apply tfplan
```

### Destroy Infrastructure

**WARNING**: This will delete all resources!

```bash
terraform destroy
```

## Troubleshooting

### Common Issues

1. **Insufficient Quota**:
   ```bash
   az vm list-usage --location norwayeast --output table
   ```

2. **AKS Node Issues**:
   ```bash
   kubectl get nodes
   kubectl describe node <node-name>
   ```

3. **Terraform State Lock**:
   ```bash
   terraform force-unlock <lock-id>
   ```

## Support

For issues or questions:
- Check Terraform documentation: https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs
- Azure support: https://azure.microsoft.com/support/
- Internal team: data-science-team@example.com

