#!/bin/bash
# Minimal Azure Deployment Script
# This creates only essential resources for testing

set -e

PROJECT_DIR="/home/saidul/Desktop/ABCD/Linkedin/Transaction-Anomaly-Detection"
cd "$PROJECT_DIR"

echo "Starting Minimal Azure Deployment"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Create Resource Group
echo -e "${BLUE}Step 1: Creating Resource Group...${NC}"
az group create \
  --name rg-aml-minimal \
  --location westeurope \
  --tags Project="AML" Owner="saidulislambinalisayed@outlook.com"

echo -e "${GREEN}Resource Group created${NC}"
echo ""

# Step 2: Create Container Registry (for Docker images)
echo -e "${BLUE}Step 2: Creating Container Registry...${NC}"
ACR_NAME="acramin$RANDOM"
az acr create \
  --name "$ACR_NAME" \
  --resource-group rg-aml-minimal \
  --location westeurope \
  --sku Basic \
  --admin-enabled true

echo -e "${GREEN}Container Registry created: $ACR_NAME${NC}"
echo ""

# Step 3: Create Storage Account
echo -e "${BLUE}Step 3: Creating Storage Account...${NC}"
STORAGE_NAME="staml$RANDOM"
az storage account create \
  --name "$STORAGE_NAME" \
  --resource-group rg-aml-minimal \
  --location westeurope \
  --sku Standard_LRS

echo -e "${GREEN}Storage Account created: $STORAGE_NAME${NC}"
echo ""

# Step 4: Create Storage Containers
echo -e "${BLUE}Step 4: Creating Storage Containers...${NC}"
STORAGE_KEY=$(az storage account keys list --resource-group rg-aml-minimal --account-name "$STORAGE_NAME" --query '[0].value' -o tsv)

az storage container create --name ml-data --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY"
az storage container create --name ml-models --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY"
az storage container create --name feature-store --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY"

echo -e "${GREEN}Storage Containers created${NC}"
echo ""

# Step 5: Build and Push Docker Image
echo -e "${BLUE}Step 5: Building Docker Image...${NC}"
cd "$PROJECT_DIR"
docker build -t transaction-anomaly-detection:latest .

echo -e "${GREEN}Docker image built${NC}"
echo ""

# Step 6: Login to ACR and Push Image
echo -e "${BLUE}Step 6: Pushing to Azure Container Registry...${NC}"
az acr login --name "$ACR_NAME"
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)

docker tag transaction-anomaly-detection:latest "$ACR_LOGIN_SERVER/transaction-anomaly-detection:latest"
docker push "$ACR_LOGIN_SERVER/transaction-anomaly-detection:latest"

echo -e "${GREEN}Image pushed to ACR${NC}"
echo ""

# Step 7: Create App Service Plan (cheapest option for web hosting)
echo -e "${BLUE}Step 7: Creating App Service (Web App)...${NC}"
az appservice plan create \
  --name asp-aml \
  --resource-group rg-aml-minimal \
  --location westeurope \
  --is-linux \
  --sku B1

# Create Web App
az webapp create \
  --name "aml-api-$RANDOM" \
  --resource-group rg-aml-minimal \
  --plan asp-aml \
  --deployment-container-image-name "$ACR_LOGIN_SERVER/transaction-anomaly-detection:latest"

# Configure Web App to use ACR
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query passwords[0].value -o tsv)

az webapp config container set \
  --name "aml-api-$RANDOM" \
  --resource-group rg-aml-minimal \
  --docker-custom-image-name "$ACR_LOGIN_SERVER/transaction-anomaly-detection:latest" \
  --docker-registry-server-url "https://$ACR_LOGIN_SERVER" \
  --docker-registry-server-user "$ACR_USERNAME" \
  --docker-registry-server-password "$ACR_PASSWORD"

echo -e "${GREEN}Web App created${NC}"
echo ""

# Save deployment info
cat > "$PROJECT_DIR/deployment_info.txt" << EOF
==================================
Minimal Azure Deployment Complete
==================================

Resource Group: rg-aml-minimal
Container Registry: $ACR_NAME
Login Server: $ACR_LOGIN_SERVER
Storage Account: $STORAGE_NAME

Next Steps:
1. View resources: https://portal.azure.com/#@/resource/subscriptions/916798ca-14df-467f-9f2a-4d37dfefee0c/resourceGroups/rg-aml-minimal
2. Access Web App: Check Azure Portal for the URL
3. View logs: az webapp log tail --name <webapp-name> --resource-group rg-aml-minimal

Estimated Cost: ~$15-20/month (Basic tier services)

To deploy full infrastructure with AKS/Databricks:
cd terraform && ../bin/terraform plan
EOF

cat "$PROJECT_DIR/deployment_info.txt"

echo ""
echo -e "${GREEN}Deployment Complete!${NC}"
echo "Check deployment_info.txt for details"

