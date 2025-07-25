#!/bin/bash

# E-Commerce Business Insights Dashboard Launcher
# This script uses 'python' command instead of 'python3'

echo "🚀 Starting E-Commerce Business Insights Dashboard..."
echo "📊 Using python command for compatibility"
echo "🌐 The app will open in your browser automatically"
echo "⏹️  Press Ctrl+C to stop the server"
echo "================================================"

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "❌ 'python' command not found!"
    echo "💡 Try creating an alias: alias python=python3"
    echo "💡 Or install Python with 'python' command available"
    exit 1
fi

# Check Python version
echo "🐍 Python version:"
python --version

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating requirements..."
    python -m pip install -r requirements.txt --user --break-system-packages
fi

# Run the Streamlit app
echo "🎯 Launching dashboard..."
python -m streamlit run business_insights_app.py
