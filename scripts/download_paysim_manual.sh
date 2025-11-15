#!/bin/bash
# Manual download script for PaySim dataset
# If Kaggle API doesn't work, use this guide

echo "============================================================"
echo "PAYSIM DATASET MANUAL DOWNLOAD GUIDE"
echo "============================================================"
echo ""
echo "Option 1: Download from Kaggle Website"
echo "  1. Go to: https://www.kaggle.com/datasets/ntnu-testimon/paysim1"
echo "  2. Click 'Download' button (requires Kaggle account)"
echo "  3. Accept dataset terms if prompted"
echo "  4. Extract the CSV file"
echo "  5. Place it in: data/PS_20174392719_1491204439457_log.csv"
echo ""
echo "Option 2: Use Kaggle API (if you have accepted terms)"
echo "  kaggle datasets download -d ntnu-testimon/paysim1 -p data --unzip"
echo ""
echo "Option 3: Alternative PaySim dataset"
echo "  kaggle datasets download -d ealaxi/paysim1 -p data --unzip"
echo ""
echo "After downloading, rename the main CSV file to:"
echo "  mv data/PS_*.csv data/transactions.csv"
echo ""
echo "============================================================"

