#!/usr/bin/env python
"""
Simple script to run the Streamlit app using python command
"""
import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit app"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "business_insights_app.py")
        
        print("🚀 Starting E-Commerce Business Insights Dashboard...")
        print(f"📍 App location: {app_path}")
        print("🌐 The app will open in your browser automatically")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run streamlit with python
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
        
    except KeyboardInterrupt:
        print("\n👋 App stopped successfully!")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        print("💡 Try installing streamlit first: python -m pip install streamlit")

if __name__ == "__main__":
    run_streamlit_app()
