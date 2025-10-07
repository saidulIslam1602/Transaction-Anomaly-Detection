"""
Setup script for Azure Databricks workspace.

This script:
1. Creates Databricks workspace configuration
2. Sets up secret scopes
3. Creates job definitions
4. Deploys notebooks
"""

import requests
import json
import os
from typing import Dict, List

class DatabricksSetup:
    """Setup Databricks workspace for AML project."""
    
    def __init__(self, workspace_url: str, token: str):
        """
        Initialize Databricks setup.
        
        Args:
            workspace_url: Databricks workspace URL
            token: Personal access token
        """
        self.workspace_url = workspace_url.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def create_secret_scope(self, scope_name: str, key_vault_uri: str) -> None:
        """
        Create secret scope backed by Azure Key Vault.
        
        Args:
            scope_name: Name of the secret scope
            key_vault_uri: Azure Key Vault URI
        """
        url = f"{self.workspace_url}/api/2.0/secrets/scopes/create"
        
        payload = {
            "scope": scope_name,
            "scope_backend_type": "AZURE_KEYVAULT",
            "backend_azure_keyvault": {
                "resource_id": key_vault_uri,
                "dns_name": key_vault_uri
            }
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            print(f"✓ Secret scope '{scope_name}' created")
        else:
            print(f"✗ Error creating secret scope: {response.text}")
    
    def upload_notebook(self, local_path: str, workspace_path: str) -> None:
        """
        Upload notebook to Databricks workspace.
        
        Args:
            local_path: Local path to notebook
            workspace_path: Workspace path for notebook
        """
        url = f"{self.workspace_url}/api/2.0/workspace/import"
        
        with open(local_path, 'r') as f:
            content = f.read()
        
        payload = {
            "path": workspace_path,
            "format": "SOURCE",
            "language": "PYTHON",
            "content": content,
            "overwrite": True
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            print(f"✓ Notebook uploaded: {workspace_path}")
        else:
            print(f"✗ Error uploading notebook: {response.text}")
    
    def create_cluster(self, cluster_config: Dict) -> str:
        """
        Create or update Databricks cluster.
        
        Args:
            cluster_config: Cluster configuration dictionary
            
        Returns:
            Cluster ID
        """
        url = f"{self.workspace_url}/api/2.0/clusters/create"
        
        response = requests.post(url, headers=self.headers, json=cluster_config)
        
        if response.status_code == 200:
            cluster_id = response.json()['cluster_id']
            print(f"✓ Cluster created: {cluster_id}")
            return cluster_id
        else:
            print(f"✗ Error creating cluster: {response.text}")
            return None
    
    def create_job(self, job_config: Dict) -> str:
        """
        Create Databricks job.
        
        Args:
            job_config: Job configuration dictionary
            
        Returns:
            Job ID
        """
        url = f"{self.workspace_url}/api/2.0/jobs/create"
        
        response = requests.post(url, headers=self.headers, json=job_config)
        
        if response.status_code == 200:
            job_id = response.json()['job_id']
            print(f"✓ Job created: {job_id}")
            return job_id
        else:
            print(f"✗ Error creating job: {response.text}")
            return None


def get_cluster_config(cluster_name: str) -> Dict:
    """Get standard cluster configuration."""
    return {
        "cluster_name": cluster_name,
        "spark_version": "13.3.x-scala2.12",
        "node_type_id": "Standard_DS3_v2",
        "driver_node_type_id": "Standard_DS3_v2",
        "autoscale": {
            "min_workers": 2,
            "max_workers": 8
        },
        "spark_conf": {
            "spark.databricks.delta.preview.enabled": "true",
            "spark.databricks.delta.optimizeWrite.enabled": "true",
            "spark.databricks.delta.autoCompact.enabled": "true"
        },
        "azure_attributes": {
            "first_on_demand": 1,
            "availability": "SPOT_WITH_FALLBACK_AZURE",
            "spot_bid_max_price": -1
        },
        "autotermination_minutes": 60,
        "enable_elastic_disk": True
    }


def get_pipeline_job_config(cluster_id: str) -> Dict:
    """Get data pipeline job configuration."""
    return {
        "name": "transaction-anomaly-detection-pipeline",
        "existing_cluster_id": cluster_id,
        "email_notifications": {
            "on_failure": ["data-team@example.com"],
            "on_success": ["data-team@example.com"]
        },
        "timeout_seconds": 7200,
        "max_retries": 2,
        "schedule": {
            "quartz_cron_expression": "0 0 2 * * ?",  # Daily at 2 AM
            "timezone_id": "Europe/Oslo",
            "pause_status": "UNPAUSED"
        },
        "tasks": [
            {
                "task_key": "ingest_data",
                "notebook_task": {
                    "notebook_path": "/Shared/transaction-anomaly-detection/01_data_ingestion",
                    "base_parameters": {}
                }
            },
            {
                "task_key": "engineer_features",
                "depends_on": [{"task_key": "ingest_data"}],
                "notebook_task": {
                    "notebook_path": "/Shared/transaction-anomaly-detection/02_feature_engineering",
                    "base_parameters": {}
                }
            },
            {
                "task_key": "train_model",
                "depends_on": [{"task_key": "engineer_features"}],
                "notebook_task": {
                    "notebook_path": "/Shared/transaction-anomaly-detection/03_model_training",
                    "base_parameters": {}
                }
            }
        ]
    }


def main():
    """Main setup function."""
    # Get configuration from environment
    workspace_url = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    key_vault_uri = os.getenv("AZURE_KEY_VAULT_URI")
    
    if not all([workspace_url, token]):
        print("Error: Missing required environment variables")
        print("Please set: DATABRICKS_HOST, DATABRICKS_TOKEN")
        return
    
    setup = DatabricksSetup(workspace_url, token)
    
    print("=" * 60)
    print("Setting up Databricks Workspace")
    print("=" * 60)
    
    # Create secret scope
    if key_vault_uri:
        print("\n1. Creating secret scope...")
        setup.create_secret_scope("aml-secrets", key_vault_uri)
    
    # Upload notebooks
    print("\n2. Uploading notebooks...")
    notebooks = [
        ("notebooks/01_data_ingestion.py", "/Shared/transaction-anomaly-detection/01_data_ingestion"),
        ("notebooks/02_feature_engineering.py", "/Shared/transaction-anomaly-detection/02_feature_engineering"),
        ("notebooks/03_model_training.py", "/Shared/transaction-anomaly-detection/03_model_training")
    ]
    
    for local_path, workspace_path in notebooks:
        if os.path.exists(local_path):
            setup.upload_notebook(local_path, workspace_path)
    
    # Create cluster
    print("\n3. Creating compute cluster...")
    cluster_config = get_cluster_config("aml-cluster")
    cluster_id = setup.create_cluster(cluster_config)
    
    # Create job
    if cluster_id:
        print("\n4. Creating scheduled job...")
        job_config = get_pipeline_job_config(cluster_id)
        job_id = setup.create_job(job_config)
    
    print("\n" + "=" * 60)
    print("Databricks setup complete!")
    print("=" * 60)
    print(f"\nAccess your workspace at: {workspace_url}")
    print("Notebooks are in: /Shared/transaction-anomaly-detection/")


if __name__ == "__main__":
    main()

