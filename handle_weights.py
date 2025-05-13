"""Handle pretrained weights for emotion recognition models."""

import os
import sys
import torch
import timm
try:
    # In Python 3.9+, use the standard collections.abc
    from collections.abc import Mapping
except ImportError:
    # In older versions, it's directly in collections
    try:
        from collections import Mapping
    except ImportError:
        # For very recent versions where Mapping is removed altogether
        Mapping = dict
import requests
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Handle pretrained weights for emotion recognition')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0.ra_in1k',
                       help='Backbone model name')
    parser.add_argument('--use_timm_pretrained', action='store_true',
                       help='Use timm pretrained weights instead of AffectNet weights')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the weights')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token for authentication')
    parser.add_argument('--force_redownload', action='store_true',
                       help='Force redownload even if weights exist')
    parser.add_argument('--class_distribution', nargs='+', type=float, default=None,
                       help='Class distribution percentages (space separated)')
    parser.add_argument('--custom_bias', type=str, default=None,
                       help='Custom bias values as comma-separated class:value pairs (e.g. "1:-2.0,3:0.5,6:0.2")')
    return parser.parse_args()

def initialize_timm_pretrained(backbone, output_path=None):
    """Initialize from timm pretrained weights for the given backbone."""
    print(f"Initializing from timm pretrained weights for {backbone}")
    
    try:
        # Check if the model is available in timm
        if backbone in timm.list_models(pretrained=True):
            # Create the model with pretrained weights
            model = timm.create_model(backbone, pretrained=True)
            
            # Extract state dict
            state_dict = model.state_dict()
            
            # Save the weights
            if output_path is None:
                output_path = f'./models/timm_pretrained_{backbone}.pth'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(state_dict, output_path)
            
            print(f"Successfully saved timm pretrained weights to {output_path}")
            return True
        else:
            print(f"Error: {backbone} not available in timm with pretrained weights")
            # List some available models as suggestions
            available_models = [m for m in timm.list_models(pretrained=True) 
                              if backbone.split('.')[0] in m][:5]
            if available_models:
                print(f"Try one of these instead: {', '.join(available_models)}")
            return False
    except Exception as e:
        print(f"Error initializing from timm: {e}")
        return False

def try_download_affectnet_weights(backbone, output_path=None, hf_token=None):
    """Try to download AffectNet pretrained weights for the given backbone."""
    # Match the backbone name to the available URL - these may require authentication
    # NOTE: These URLs are currently not working (404)
    # Keeping this function for future use if URLs become available again
    print('AffectNet pretrained weights are currently unavailable.')
    print('Using timm pretrained weights instead.')
    return False

def verify_weights(file_path):
    """Verify that weights file can be loaded correctly."""
    try:
        # Try to load the weights
        state_dict = torch.load(file_path, map_location='cpu')
        
        # Check that it's a valid state dict or dict with state_dict
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Count the number of parameters
            param_count = sum(v.numel() for v in state_dict.values())
            print(f"Weights file verified: contains {param_count} parameters")
            
            # If we get this far, the file is valid
            return True
    except Exception as e:
        print(f"Error verifying weights file: {e}")
    
    return False

def initialize_classifier_weights(weights_path, bias_correction=True, class_distribution=None, custom_bias_values=None):
    """Re-initialize the classifier weights.
    
    Args:
        weights_path: Path to the weights file
        bias_correction: Whether to apply bias correction
        class_distribution: Optional list/array of class distribution percentages
        custom_bias_values: Optional dictionary mapping class indices to bias values
    """
    try:
        # Load the weights
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle different formats
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove any module. prefix
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v
        
        # Reinitialize classifier weights
        classifier_keys = [k for k in clean_state_dict.keys() if any(x in k for x in ['classifier', 'fc', 'head', 'output'])]
        if classifier_keys:
            print(f"Found classifier keys: {classifier_keys}")
            for key in classifier_keys:
                if 'weight' in key:
                    shape = clean_state_dict[key].shape
                    print(f"Reinitializing classifier weights with shape {shape}")
                    # Initialize with Kaiming normal
                    clean_state_dict[key] = torch.nn.init.kaiming_normal_(
                        torch.zeros(shape), 
                        mode='fan_out', 
                        nonlinearity='relu'
                    )
                    
                    # EXTREME MEASURE: Zero out disgust class weights completely
                    if len(shape) == 2 and shape[0] >= 7:  # At least 7 emotion classes
                        # Zero all weights for disgust class (class index 1)
                        clean_state_dict[key][1, :] = 0.0
                        # Make other class weights stronger
                        clean_state_dict[key][3, :] *= 1.5  # Happy
                        clean_state_dict[key][6, :] *= 1.3  # Neutral
                        print(f"EXTREME: Zeroed out ALL weights for disgust class")
                        
                if 'bias' in key and bias_correction:
                    # Initialize biases with a class-balanced approach
                    bias_shape = clean_state_dict[key].shape[0]
                    
                    if bias_shape == 7:  # 7 emotion classes
                        print("Applying emotion-specific bias correction")
                        
                        # Default bias values for FER2013 and similar datasets
                        bias_values = torch.zeros(bias_shape)
                        
                        if custom_bias_values is not None:
                            # Use custom bias values if provided
                            print("Using custom bias values")
                            for idx, value in custom_bias_values.items():
                                if 0 <= idx < bias_shape:
                                    bias_values[idx] = value
                        elif class_distribution is not None:
                            # Calculate bias based on class distribution
                            print("Calculating bias values from class distribution")
                            if len(class_distribution) == bias_shape:
                                # Convert percentages to log space for bias
                                total = sum(class_distribution)
                                class_percentages = [count/total for count in class_distribution]
                                mean_percentage = 1.0 / bias_shape
                                
                                # Set bias proportional to log of class imbalance
                                # Negative for over-represented classes, positive for under-represented
                                for i in range(bias_shape):
                                    ratio = mean_percentage / max(class_percentages[i], 0.001)
                                    # Scale and clip the bias value
                                    bias_values[i] = min(max(0.5 * torch.log(torch.tensor(ratio)), -2.0), 2.0)
                            else:
                                print(f"Warning: class_distribution length {len(class_distribution)} does not match bias shape {bias_shape}")
                        else:
                            # Use default hardcoded values with EXTREME adjustments for known issues
                            print("Using EXTREME bias values to block disgust predictions")
                            bias_values[0] = 1.0      # angry: positive bias
                            bias_values[1] = -50.0    # disgust: EXTREME negative - effectively disable this class
                            bias_values[2] = 0.0      # fear: neutral
                            bias_values[3] = 5.0      # happy: strong positive
                            bias_values[4] = 1.0      # sad: positive bias
                            bias_values[5] = 0.5      # surprise: small positive
                            bias_values[6] = 3.0      # neutral: strong positive
                        
                        clean_state_dict[key] = bias_values
                        print(f"Applied extreme bias correction: {bias_values}")
                    else:
                        # For other shapes, just zero the bias
                        clean_state_dict[key].zero_()
        
        # Save the fixed weights
        torch.save(clean_state_dict, weights_path)
        print("Successfully reinitialized classifier weights with EXTREME bias correction")
        return True
    except Exception as e:
        print(f"Error fixing weights: {e}")
        return False

def prepare_weights(backbone, use_timm_pretrained=False, output_path=None, hf_token=None, force_redownload=False, 
                    class_distribution=None, custom_bias_values=None):
    """Prepare weights for the given backbone."""
    if output_path is None:
        if use_timm_pretrained:
            output_path = f"./models/timm_pretrained_{backbone}.pth"
        else:
            output_path = f"./models/affectnet_pretrained_{backbone}.pth"
    
    # Check if we already have valid weights before trying to download
    if os.path.exists(output_path) and not force_redownload:
        print(f"Checking existing weights at {output_path}")
        if verify_weights(output_path):
            print("Existing weights are valid.")
            # Reinitialize the classifier
            if initialize_classifier_weights(output_path, class_distribution=class_distribution, 
                                            custom_bias_values=custom_bias_values):
                print("Using existing weights file with updated classifier bias")
                return True
            else:
                # If we couldn't fix the weights, use a different approach
                print("Could not update classifier weights, will try to get new weights")
                os.remove(output_path)
        else:
            # Invalid file, remove it
            print("Existing weights are invalid. Removing...")
            os.remove(output_path)
    elif os.path.exists(output_path) and force_redownload:
        print(f"Force redownload requested - removing existing weights at {output_path}")
        os.remove(output_path)
    
    # Skip HuggingFace downloads completely and use timm pretrained weights directly
    print("Using timm pretrained weights directly")
    if initialize_timm_pretrained(backbone, output_path):
        # Reinitialize classifier weights
        if initialize_classifier_weights(output_path, class_distribution=class_distribution,
                                       custom_bias_values=custom_bias_values):
            return True
    
    return False

if __name__ == "__main__":
    args = parse_args()
    
    output_path = args.output_path
    if output_path is None:
        if args.use_timm_pretrained:
            output_path = f"./models/timm_pretrained_{args.backbone}.pth"
        else:
            output_path = f"./models/affectnet_pretrained_{args.backbone}.pth"
    
    # Check if token is in environment variable if not provided as argument
    hf_token = args.hf_token
    if hf_token is None and 'HF_TOKEN' in os.environ:
        hf_token = os.environ['HF_TOKEN']
        print("Using HuggingFace token from environment variable")
        print("Note: HuggingFace downloads are disabled, using timm weights only")
    
    # Process custom bias values if provided
    custom_bias_values = None
    if args.custom_bias:
        custom_bias_values = {}
        # Parse the comma-separated string of class:bias pairs
        bias_pairs = args.custom_bias.split(',')
        for pair in bias_pairs:
            if ':' in pair:
                idx, val = pair.split(':')
                try:
                    custom_bias_values[int(idx)] = float(val)
                except ValueError:
                    print(f"Warning: Could not parse bias pair {pair}. Skipping.")
        print(f"Using custom bias values: {custom_bias_values}")
    
    # Only force redownload if explicitly requested, not just because HF token is provided
    if prepare_weights(args.backbone, True, output_path, None, 
                      args.force_redownload, args.class_distribution, custom_bias_values):
        print("Weights preparation successful")
        sys.exit(0)
    else:
        print("Weights preparation failed")
        sys.exit(1) 