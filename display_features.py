# Feature Information Display Script
import sys
import os

# Add project root to Python path dynamically
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Note: This script expects the notebook variables to be available
# Run this after executing the notebook cells

print('='*80)
print('FEATURE INFORMATION - CUSTOMER SUMMARY')
print('='*80)
