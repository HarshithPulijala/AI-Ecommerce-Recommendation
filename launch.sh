#!/bin/bash
# Launch script for AI Recommendation Engine Web App

echo "🚀 AI Recommendation Engine - Local Development Launch"
echo "======================================================"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📚 Ensuring dependencies are installed..."
pip install -q -r requirements.txt

# Export Flask app
export FLASK_APP=src.app
export FLASK_ENV=development

# Create startup message
echo ""
echo "✅ Environment ready!"
echo ""
echo "Available commands:"
echo "  - Start Flask server: python -m flask run --port 5000"
echo "  - Run tests: python test_webapp.py"
echo "  - Open browser: http://localhost:5000"
echo ""
echo "======================================================"
