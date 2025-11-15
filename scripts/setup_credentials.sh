#!/bin/bash
# Credentials Setup Script
# This script helps you set up environment variables for the project

set -e

echo "🔐 Transaction Anomaly Detection - Credentials Setup"
echo "======================================================"
echo ""

# Check if .env file exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ ! $overwrite =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file. Exiting."
        exit 0
    fi
fi

echo "This script will help you set up your credentials."
echo "You can skip any step by pressing Enter."
echo ""

# Create .env file
cat > .env << 'EOF'
# Transaction Anomaly Detection - Environment Variables
# DO NOT COMMIT THIS FILE TO VERSION CONTROL
# This file is already in .gitignore

EOF

# Databricks Configuration
echo "📊 Databricks Configuration"
echo "---------------------------"
read -p "Databricks Host (e.g., adb-1234567890123456.7.azuredatabricks.net): " databricks_host
if [ ! -z "$databricks_host" ]; then
    echo "DATABRICKS_HOST=$databricks_host" >> .env
fi

read -p "Databricks HTTP Path (e.g., /sql/1.0/warehouses/abc123def456): " databricks_path
if [ ! -z "$databricks_path" ]; then
    echo "DATABRICKS_HTTP_PATH=$databricks_path" >> .env
fi

read -p "Databricks Token (dapi...): " databricks_token
if [ ! -z "$databricks_token" ]; then
    echo "DATABRICKS_TOKEN=$databricks_token" >> .env
fi

echo ""

# Kafka Configuration
echo "📨 Kafka Configuration (optional)"
echo "----------------------------------"
read -p "Kafka Bootstrap Servers (e.g., localhost:9092 or kafka.example.com:9092): " kafka_servers
if [ ! -z "$kafka_servers" ]; then
    echo "REAL_TIME_KAFKA_BOOTSTRAP_SERVERS=$kafka_servers" >> .env
fi

echo ""

# Redis Configuration
echo "💾 Redis Configuration (optional)"
echo "----------------------------------"
read -p "Redis Host (e.g., localhost or redis.example.com): " redis_host
if [ ! -z "$redis_host" ]; then
    echo "REAL_TIME_REDIS_HOST=$redis_host" >> .env
fi

read -p "Redis Port (default: 6379): " redis_port
if [ ! -z "$redis_port" ]; then
    echo "REAL_TIME_REDIS_PORT=$redis_port" >> .env
fi

echo ""

# Email Configuration
echo "📧 Email Configuration (optional)"
echo "-----------------------------------"
read -p "SMTP Server (e.g., smtp.gmail.com): " smtp_server
if [ ! -z "$smtp_server" ]; then
    echo "NOTIFICATIONS_EMAIL_SMTP_SERVER=$smtp_server" >> .env
fi

read -p "From Address (e.g., alerts@yourdomain.com): " from_address
if [ ! -z "$from_address" ]; then
    echo "NOTIFICATIONS_EMAIL_FROM_ADDRESS=$from_address" >> .env
fi

echo ""

# Webhook Configuration
echo "🔗 Webhook Configuration (optional)"
echo "------------------------------------"
read -p "Webhook URL (e.g., https://your-webhook.com/endpoint): " webhook_url
if [ ! -z "$webhook_url" ]; then
    echo "NOTIFICATIONS_WEBHOOK_URL=$webhook_url" >> .env
fi

echo ""

# OpenAI Configuration
echo "🤖 OpenAI Configuration (optional)"
echo "-----------------------------------"
read -p "OpenAI API Key (sk-...): " openai_key
if [ ! -z "$openai_key" ]; then
    echo "OPENAI_API_KEY=$openai_key" >> .env
fi

echo ""
echo "✅ Credentials setup complete!"
echo ""
echo "📝 Created .env file with your credentials."
echo "🔒 This file is in .gitignore and will NOT be committed."
echo ""
echo "💡 To use these credentials:"
echo "   1. Source the file: source .env"
echo "   2. Or restart your terminal/shell"
echo "   3. Or use: export \$(cat .env | xargs)"
echo ""
echo "📖 See CREDENTIALS_SETUP.md for more details."

