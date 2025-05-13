#!/usr/bin/env python3

"""
Download AffectNet pretrained weights for all supported backbone architectures.
This script should be run before training to ensure all pretrained weights are available.
"""

import os
import requests
from tqdm import tqdm
import argparse

def download_file(url, target_path):
    """Download file from URL with progress bar."""
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Show download progress
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(target_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        
        if total_size != 0 and progress_bar.n != total_size:
            print(f"ERROR: Downloaded size doesn't match expected size for {target_path}")
            if os.path.exists(target_path):
                os.remove(target_path)
            return False
        
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Download AffectNet pretrained weights')
    parser.add_argument('--backbone', choices=['all', 'efficientnet_b0', 'resnet50', 'swin_v2_b', 'vit_base_patch16_224'],
                        default='all', help='Backbone architecture to download weights for')
    parser.add_argument('--output_dir', type=str, default='models', 
                        help='Directory to save pretrained weights')
    args = parser.parse_args()

    # Map backbones to available pretrained model URLs
    pretrained_urls = {
        'efficientnet_b0': 'https://huggingface.co/datasets/jamescalam/affectnet-faces/resolve/main/models/efficientnet_b0_affectnet.pth',
        'resnet50': 'https://huggingface.co/datasets/jamescalam/affectnet-faces/resolve/main/models/resnet50_affectnet.pth',
        'swin_v2_b': 'https://huggingface.co/meowitsavor/swin-base-AffectNet8-emotions-81-09/resolve/main/pytorch_model.bin',
        'vit_base_patch16_224': 'https://huggingface.co/datasets/jamescalam/affectnet-faces/resolve/main/models/vit_affectnet.pth',
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which backbones to download
    backbones = list(pretrained_urls.keys()) if args.backbone == 'all' else [args.backbone]
    
    # Download each backbone's weights
    for backbone in backbones:
        if backbone not in pretrained_urls:
            print(f"No pretrained weights available for {backbone}")
            continue
            
        url = pretrained_urls[backbone]
        target_path = os.path.join(args.output_dir, f"affectnet_pretrained_{backbone}.pth")
        
        # Skip if file already exists
        if os.path.exists(target_path):
            print(f"Weights for {backbone} already exist at {target_path}")
            continue
            
        print(f"Downloading weights for {backbone} from {url}")
        success = download_file(url, target_path)
        
        if success:
            print(f"Successfully downloaded weights for {backbone} to {target_path}")
        else:
            print(f"Failed to download weights for {backbone}")
    
    print("Download process complete!")

if __name__ == "__main__":
    main() 