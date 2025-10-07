#!/bin/bash

# Terraform Setup Script
# Downloads and installs Terraform for this project

set -e

echo "🔧 Setting up Terraform for Azure deployment..."

# Create bin directory if it doesn't exist
mkdir -p bin

# Download Terraform if not exists
if [ ! -f "bin/terraform" ]; then
    echo "📥 Downloading Terraform..."
    cd bin
    wget https://releases.hashicorp.com/terraform/1.6.6/terraform_1.6.6_linux_amd64.zip
    unzip terraform_1.6.6_linux_amd64.zip
    rm terraform_1.6.6_linux_amd64.zip
    chmod +x terraform
    cd ..
    echo "✅ Terraform installed successfully"
else
    echo "✅ Terraform already installed"
fi

# Initialize Terraform
echo "🚀 Initializing Terraform..."
./bin/terraform init

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy terraform.tfvars.example to terraform.tfvars"
echo "2. Edit terraform.tfvars with your values"
echo "3. Run: ./bin/terraform plan"
echo "4. Run: ./bin/terraform apply"
echo ""
echo "For minimal deployment, use: ../deploy_minimal.sh"
