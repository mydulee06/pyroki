#!/usr/bin/env python3
"""
JSON to PT Converter for CMA-ES
Converts inverse results JSON files to PT format compatible with cma_es.py
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
import sys
import traceback

def load_inverse_results(json_file: str) -> List[Dict[str, Any]]:
    """Load inverse results from JSON file with error handling"""
    try:
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list of samples")
        
        print(f"✅ Loaded {len(data)} total samples from {json_file}")
        return data
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")

def validate_sample(sample: Dict[str, Any], index: int) -> bool:
    """Validate if a sample has required fields"""
    required_fields = ['sampled_x', 'sampled_y', 'sampled_yaw']
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f"⚠️  Warning: Sample {index} missing fields: {missing_fields}")
        return False
    
    # Check if values are numeric
    for field in required_fields:
        value = sample[field]
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            print(f"⚠️  Warning: Sample {index} has invalid {field}: {value}")
            return False
    
    return True

def extract_xyyaw_data(inverse_results: List[Dict[str, Any]], 
                      pos_threshold: float = 0.02,
                      ori_threshold: float = 0.1,
                      col_threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract x, y, yaw samples and success data from inverse results
    
    Args:
        inverse_results: List of sample dictionaries
        pos_threshold: Position error threshold (default: 0.02m = 2cm)
        ori_threshold: Orientation error threshold (default: 0.1 rad ≈ 5.7°)
        col_threshold: Collision cost threshold (default: 0.001)
    
    Returns:
        xyyaw_samples: numpy array of shape (N, 3) with [x, y, yaw] for each sample
        success: numpy array of shape (N,) with boolean success values
        sample_range: dict with x, y, yaw ranges
    """
    if not inverse_results:
        raise ValueError("No samples available")
    
    valid_samples = []
    valid_indices = []
    
    # Filter valid samples
    for i, sample in enumerate(inverse_results):
        if validate_sample(sample, i):
            valid_samples.append(sample)
            valid_indices.append(i)
    
    if not valid_samples:
        raise ValueError("No valid samples found")
    
    print(f"📊 Using {len(valid_samples)}/{len(inverse_results)} valid samples")
    
    # Extract x, y, yaw values and success status
    xyyaw_samples = []
    success = []
    
    for sample in valid_samples:
        x = float(sample['sampled_x'])
        y = float(sample['sampled_y'])
        yaw = float(sample['sampled_yaw'])
        
        xyyaw_samples.append([x, y, yaw])
        
        # Determine success based on error thresholds
        max_pos_error = sample.get('max_position_error', float('inf'))
        max_ori_error = sample.get('max_orientation_error', float('inf'))
        max_col_cost = sample.get('max_collision_cost', float('inf'))
        
        # Convert to float if they exist, otherwise use inf (failure)
        max_pos_error = float(max_pos_error) if max_pos_error is not None else float('inf')
        max_ori_error = float(max_ori_error) if max_ori_error is not None else float('inf')
        max_col_cost = float(max_col_cost) if max_col_cost is not None else float('inf')
        
        is_success = (max_pos_error < pos_threshold and 
                     max_ori_error < ori_threshold and 
                     max_col_cost < col_threshold)
        
        success.append(is_success)
    
    # Convert to numpy arrays
    xyyaw_samples = np.array(xyyaw_samples, dtype=np.float32)
    success = np.array(success, dtype=bool)
    
    # Calculate sample ranges
    x_range = [float(xyyaw_samples[:, 0].min()), float(xyyaw_samples[:, 0].max())]
    y_range = [float(xyyaw_samples[:, 1].min()), float(xyyaw_samples[:, 1].max())]
    yaw_range = [float(xyyaw_samples[:, 2].min()), float(xyyaw_samples[:, 2].max())]
    
    sample_range = {
        'x': x_range,
        'y': y_range,
        'yaw': yaw_range
    }
    
    # Print statistics
    print(f"📈 Sample ranges:")
    print(f"   - x: [{x_range[0]:.4f}, {x_range[1]:.4f}] (range: {x_range[1]-x_range[0]:.4f})")
    print(f"   - y: [{y_range[0]:.4f}, {y_range[1]:.4f}] (range: {y_range[1]-y_range[0]:.4f})")
    print(f"   - yaw: [{yaw_range[0]:.4f}, {yaw_range[1]:.4f}] (range: {yaw_range[1]-yaw_range[0]:.4f})")
    print(f"🎯 Success rate: {success.sum()}/{len(success)} ({success.mean():.2%})")
    
    return xyyaw_samples, success, sample_range

def convert_json_to_pt(json_file: str, 
                      output_file: str,
                      pos_threshold: float = 0.02,
                      ori_threshold: float = 0.1,
                      col_threshold: float = 0.001):
    """
    Convert JSON inverse results to PT format for CMA-ES
    
    Args:
        json_file: Path to input JSON file
        output_file: Path to output PT file
        pos_threshold: Position error threshold for success determination
        ori_threshold: Orientation error threshold for success determination
        col_threshold: Collision cost threshold for success determination
    """
    print(f"🔄 Converting {json_file} to {output_file}")
    print(f"📏 Thresholds: pos={pos_threshold}m, ori={ori_threshold}rad, col={col_threshold}")
    
    try:
        # Load inverse results
        inverse_results = load_inverse_results(json_file)
        
        # Extract data
        xyyaw_samples, success, sample_range = extract_xyyaw_data(
            inverse_results, pos_threshold, ori_threshold, col_threshold
        )
        
        # Convert numpy arrays to torch tensors for consistency
        xyyaw_tensor = torch.from_numpy(xyyaw_samples)
        success_tensor = torch.from_numpy(success)
        
        # Create data dictionary compatible with cma_es.py
        import time
        data = {
            'sample_range': sample_range,
            'xyyaw_samples': xyyaw_tensor,     # torch tensor - all samples
            'success': success_tensor,         # torch tensor - all samples
            'num_samples': len(xyyaw_samples),
            'num_successful': int(success.sum()),
            'success_rate': float(success.mean()),
            'conversion_info': {
                'source_file': str(json_file),
                'conversion_timestamp': time.time(),  # Unix timestamp
                'thresholds': {
                    'position': pos_threshold,
                    'orientation': ori_threshold,
                    'collision': col_threshold
                }
            }
        }
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Debug: Print data types before saving
        print(f"🔍 Debug info before saving:")
        print(f"   xyyaw_samples type: {type(xyyaw_tensor)}, shape: {xyyaw_tensor.shape}, dtype: {xyyaw_tensor.dtype}")
        print(f"   success type: {type(success_tensor)}, shape: {success_tensor.shape}, dtype: {success_tensor.dtype}")
        
        # Save as PT file using torch with explicit settings
        try:
            torch.save(data, output_file, pickle_protocol=2)  # Use older pickle protocol for compatibility
            print(f"✅ Successfully saved PT file to {output_file}")
        except Exception as e:
            print(f"❌ Error saving with torch.save: {e}")
            # Fallback: try saving with numpy
            print("🔄 Trying fallback save with numpy...")
            fallback_data = {
                'sample_range': sample_range,
                'xyyaw_samples': xyyaw_samples,  # numpy array
                'success': success,              # numpy array  
                'num_samples': len(xyyaw_samples),
                'num_successful': int(success.sum()),
                'success_rate': float(success.mean()),
                'conversion_info': {
                    'source_file': str(json_file),
                    'conversion_timestamp': time.time(),
                    'thresholds': {
                        'position': pos_threshold,
                        'orientation': ori_threshold,
                        'collision': col_threshold
                    }
                }
            }
            np.save(output_file.replace('.pt', '.npy'), fallback_data)
            print(f"✅ Fallback: Saved as NPY file instead")
            return
        
        print(f"   📊 {len(xyyaw_samples)} total samples")
        print(f"   🎯 {success.sum()} successful samples")
        print(f"   📈 Success rate: {success.mean():.2%}")
        print(f"   💾 File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Verify the saved file can be loaded with different methods
        print(f"🔍 Testing different loading methods:")
        
        # Test 1: torch.load
        try:
            loaded_data = torch.load(output_file)
            print(f"✅ torch.load: Success")
            print(f"   Keys: {list(loaded_data.keys())}")
            print(f"   xyyaw_samples type: {type(loaded_data['xyyaw_samples'])}")
        except Exception as e:
            print(f"❌ torch.load: Failed - {e}")
        
        # Test 2: torch.load with map_location
        try:
            loaded_data = torch.load(output_file, map_location='cpu')
            print(f"✅ torch.load(map_location='cpu'): Success")
        except Exception as e:
            print(f"❌ torch.load(map_location='cpu'): Failed - {e}")
        
        # Test 3: Try to load as numpy (this shouldn't work for .pt files)
        try:
            loaded_data = np.load(output_file, allow_pickle=True)
            print(f"⚠️  np.load: Unexpectedly succeeded - this suggests the file might not be a proper torch file")
            print(f"   Type: {type(loaded_data)}")
        except Exception as e:
            print(f"✅ np.load: Failed as expected - {e}")
        
        # Test 4: Check file header
        try:
            with open(output_file, 'rb') as f:
                header = f.read(10)
                print(f"🔍 File header (first 10 bytes): {header}")
        except Exception as e:
            print(f"❌ Could not read file header: {e}")
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON inverse results to PT format for CMA-ES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", required=True, 
                       help="Input JSON file path")
    parser.add_argument("--output", "-o", 
                       help="Output PT file path (default: input_name.pt)")
    parser.add_argument("--pos-threshold", type=float, default=0.02,
                       help="Position error threshold for success (meters)")
    parser.add_argument("--ori-threshold", type=float, default=0.1,
                       help="Orientation error threshold for success (radians)")
    parser.add_argument("--col-threshold", type=float, default=0.001,
                       help="Collision cost threshold for success")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.with_suffix('.pt')
    
    print(f"🚀 Starting conversion...")
    print(f"   📁 Input: {args.input}")
    print(f"   📁 Output: {output_path}")
    
    try:
        # Convert file
        convert_json_to_pt(
            args.input, 
            str(output_path),
            args.pos_threshold,
            args.ori_threshold,
            args.col_threshold
        )
        print(f"🎉 Conversion completed successfully!")
        
    except Exception as e:
        print(f"💥 Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()