#!/usr/bin/env python
"""
Wrapper script to train and evaluate the FER2013 model.
"""

import os
import sys

if __name__ == "__main__":
    # Add the root directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the main function
    from src.main import main
    main() 