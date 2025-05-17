#!/usr/bin/env python3
"""Check and fix all Python files in emotion_net directory for Linux compatibility."""

import os
import sys

def fix_file(filepath):
    """Fix file encoding and line endings for Linux."""
    try:
        # Read the file content as binary
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Check if file contains null bytes
        has_null_bytes = b'\x00' in content
        
        # Check if file contains Windows line endings
        has_crlf = b'\r\n' in content
        
        # If file has issues, fix them
        if has_null_bytes or has_crlf:
            print(f"Fixing {filepath}...")
            
            # Remove null bytes
            if has_null_bytes:
                print(f"  - Removing null bytes")
                content = content.replace(b'\x00', b'')
            
            # Convert Windows line endings to Unix
            if has_crlf:
                print(f"  - Converting Windows line endings to Unix")
                content = content.replace(b'\r\n', b'\n')
            
            # Write back the cleaned content
            with open(filepath, 'wb') as f:
                f.write(content)
            
            print(f"  ✅ Fixed {filepath}")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def check_directory(directory="emotion_net"):
    """Check all Python files in directory and subdirectories."""
    fixed_count = 0
    checked_count = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                checked_count += 1
                
                if fix_file(filepath):
                    fixed_count += 1
    
    return fixed_count, checked_count

if __name__ == "__main__":
    print("Checking Python files for Linux compatibility...")
    
    fixed_count, checked_count = check_directory()
    
    print(f"\nSummary: Checked {checked_count} files, fixed {fixed_count} files")
    
    if fixed_count > 0:
        print("\n✅ All issues have been fixed. Files are now Linux-compatible.")
    else:
        print("\n✅ No issues found. All files are already Linux-compatible.")
    
    print("\nReminder: When uploading to Linux, use binary/exact transfer mode to preserve line endings.") 