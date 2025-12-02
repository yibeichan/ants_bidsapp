import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Add the site-packages directory to the Python path
site_packages = os.path.join(os.path.dirname(sys.executable), 'lib', 'python3.12', 'site-packages')
if os.path.exists(site_packages):
    sys.path.insert(0, site_packages)

# Print the Python path for debugging
print(f"Python path: {sys.path}") 