#!/usr/bin/env python3
"""Fix null bytes in Python files"""

import os
import sys

def fix_file(filepath):
    try:
        # Read the file content
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Check if file contains null bytes
        if b'\x00' in content:
            print(f"Fixing null bytes in {filepath}")
            # Remove null bytes
            content = content.replace(b'\x00', b'')
            # Replace Windows line endings (CRLF) with Unix line endings (LF)
            content = content.replace(b'\r\n', b'\n')
            
            # Write back the cleaned content
            with open(filepath, 'wb') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def fix_directory(directory):
    fixed_count = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                if fix_file(filepath):
                    fixed_count += 1
    
    return fixed_count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "emotion_net"
    
    print(f"Fixing null bytes in Python files in {directory}...")
    fixed_count = fix_directory(directory)
    print(f"Fixed {fixed_count} files") 