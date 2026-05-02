#!/bin/bash

# Trail Popularity Dashboard Launcher
# Simple script to run the Streamlit app

echo "=========================================="
echo "🥾 Trail Popularity Prediction Dashboard"
echo "=========================================="
echo ""
echo "Checking Python environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Make sure you're in the dashboard directory."
    exit 1
fi

if [ ! -d "Data" ]; then
    echo "⚠️  Warning: Data directory not found"
    echo "   Make sure preprocessed_trails.csv is in ./Data/"
fi

echo "✓ Python found: $(python3 --version)"
echo ""
echo "Installing/checking dependencies..."

# Install or upgrade requirements
pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies ready"
echo ""
echo "Starting Streamlit app..."
echo "Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py

