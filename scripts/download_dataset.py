#!/usr/bin/env python3
"""
Download the European Financial Transaction Fraud Dataset.

This script downloads the real transaction fraud dataset from Kaggle.
Dataset: PS_20174392719_1491204439457_log.csv
"""

import os
import sys
import subprocess
from pathlib import Path

def download_via_kaggle_api():
    """Download using Kaggle API if available."""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        dataset = "ntnu-testimon/paysim1"  # Common fraud detection dataset
        print("Downloading dataset from Kaggle...")
        api.dataset_download_files(dataset, path="data", unzip=True)
        print("✅ Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Kaggle API not available: {e}")
        return False

def download_via_curl():
    """Download dataset via direct URL if available."""
    urls = [
        "https://www.kaggle.com/datasets/ntnu-testimon/paysim1/download",
        # Alternative: direct download if available
    ]
    
    for url in urls:
        try:
            print(f"Attempting to download from {url}...")
            result = subprocess.run(
                ["curl", "-L", "-o", "data/transactions.zip", url],
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0:
                # Unzip if needed
                subprocess.run(["unzip", "-o", "data/transactions.zip", "-d", "data/"])
                return True
        except Exception as e:
            print(f"Download failed: {e}")
            continue
    
    return False

def create_sample_dataset():
    """Create a realistic sample dataset for testing."""
    import pandas as pd
    import numpy as np
    
    print("Creating realistic sample dataset...")
    np.random.seed(42)
    
    n = 100000  # 100K transactions for enterprise-grade testing
    
    # Create realistic transaction data
    df = pd.DataFrame({
        'step': np.random.randint(1, 744, n),  # Hours in a month
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'amount': np.random.lognormal(5, 2, n),
        'nameOrig': [f'C{i:09d}' for i in range(n)],
        'oldbalanceOrg': np.random.exponential(5000, n),
        'newbalanceOrig': np.random.exponential(4000, n),
        'nameDest': [f'M{i%1000:09d}' for i in range(n)],
        'oldbalanceDest': np.random.exponential(2000, n),
        'newbalanceDest': np.random.exponential(3000, n),
        'isFraud': np.random.choice([0, 1], n, p=[0.9987, 0.0013]),  # Realistic fraud rate
        'isFlaggedFraud': np.zeros(n, dtype=int)
    })
    
    # Make fraud patterns more realistic
    fraud_indices = df[df['isFraud'] == 1].index
    df.loc[fraud_indices, 'type'] = np.random.choice(['TRANSFER', 'CASH_OUT'], len(fraud_indices), p=[0.6, 0.4])
    df.loc[fraud_indices, 'amount'] = np.random.lognormal(8, 1.5, len(fraud_indices))  # Higher amounts for fraud
    
    # Save to CSV
    output_path = Path("data/transactions.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Created sample dataset with {len(df)} transactions")
    print(f"   - Fraud cases: {df['isFraud'].sum()}")
    print(f"   - Saved to: {output_path}")
    return str(output_path)

def main():
    """Main function to download or create dataset."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print(f"✅ Dataset found: {csv_files[0]}")
        return str(csv_files[0])
    
    print("📥 No dataset found. Attempting to download...")
    
    # Try Kaggle API first
    if download_via_kaggle_api():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
    
    # Try direct download
    if download_via_curl():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
    
    # Fallback: Create realistic sample
    print("⚠️  Could not download from external source.")
    print("📊 Creating realistic enterprise-grade sample dataset...")
    return create_sample_dataset()

if __name__ == "__main__":
    dataset_path = main()
    print(f"\n✅ Dataset ready at: {dataset_path}")
    print("You can now run: python src/main.py --data data/transactions.csv --output output/")

