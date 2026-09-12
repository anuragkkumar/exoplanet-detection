import sys
import os

# Add root directory to path for Hugging Face Spaces compatibility
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Launch Streamlit dashboard logic
import app.dashboard
