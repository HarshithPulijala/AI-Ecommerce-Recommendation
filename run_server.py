"""
Simple server runner script
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the app
from src.app import app, initialize_engine

if __name__ == '__main__':
    print("Initializing recommendation engine...")
    initialize_engine()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
