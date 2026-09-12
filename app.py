import sys
import os
import runpy

# Ensure root directory is at the front of sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Execute Streamlit dashboard script in main context
dashboard_path = os.path.join(root_dir, "app", "dashboard.py")
runpy.run_path(dashboard_path, run_name="__main__")
