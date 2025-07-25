"""
This script is used to perform inverse transform on sampled values in JSON file.

Usage:
python inverse_transform_results.py <input_file> [-o <output_file>] [--verify]

Example:
python inverse_transform_results.py /path/to/input.json -o /path/to/output.json --verify
"""

import json
import numpy as np
import argparse
from pathlib import Path
import jax.numpy as jnp
import jaxlie

def inverse_transform(sampled_x, sampled_y, sampled_z, sampled_yaw):
    """
    Compute inverse of SE(3) transform using jaxlie library
    
    Original transform: T_mid_sole_welding_object
    Returns: T_welding_object_mid_sole = T_mid_sole_welding_object^(-1)
    """
    # Create SE(3) transform from original sample
    so3 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, sampled_yaw)
    T_mid_sole_welding = jaxlie.SE3.from_rotation_and_translation(
        so3, jnp.array([sampled_x, sampled_y, sampled_z])
    )
    
    # Compute inverse
    T_welding_mid_sole = T_mid_sole_welding.inverse()
    
    # Extract components
    translation = T_welding_mid_sole.translation()
    inv_x, inv_y, inv_z = float(translation[0]), float(translation[1]), float(translation[2])
    
    # Extract yaw angle from rotation
    rpy = T_welding_mid_sole.rotation().as_rpy_radians()
    inv_yaw = float(rpy[2])  # Extract yaw (Z rotation)
    
    return inv_x, inv_y, inv_z, inv_yaw


def process_json_file(input_file, output_file):
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Input file: {input_file}")
    print(f"Processing {len(data)} samples...")
    
    # Perform inverse transform
    inverse_data = []
    for i, sample in enumerate(data):
        # Extract original values
        orig_x = sample['sampled_x']
        orig_y = sample['sampled_y']
        orig_z = sample['sampled_z']
        orig_yaw = sample['sampled_yaw']
        
        # Calculate inverse transform
        inv_x, inv_y, inv_z, inv_yaw = inverse_transform(orig_x, orig_y, orig_z, orig_yaw)
        
        # Create new sample (existing info + inverse transform results)
        inverse_sample = sample.copy()
        inverse_sample['sampled_x'] = float(inv_x)
        inverse_sample['sampled_y'] = float(inv_y)
        inverse_sample['sampled_z'] = float(inv_z)
        inverse_sample['sampled_yaw'] = float(inv_yaw)
        
        # Also record original values
        inverse_sample['original_sampled_x'] = float(orig_x)
        inverse_sample['original_sampled_y'] = float(orig_y)
        inverse_sample['original_sampled_z'] = float(orig_z)
        inverse_sample['original_sampled_yaw'] = float(orig_yaw)
        
        inverse_data.append(inverse_sample)
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{len(data)} samples processed")
    
    # Save results to new JSON file
    with open(output_file, 'w') as f:
        json.dump(inverse_data, f, indent=2)
    
    print(f"Inverse transform completed! Results saved to {output_file}")
    
    # Show some examples
    print("\n=== Inverse Transform Examples (First 3 samples) ===")
    for i in range(min(3, len(data))):
        sample = data[i]
        inv_sample = inverse_data[i]
        print(f"\nSample {i+1}:")
        print(f"  Original:  x={sample['sampled_x']:.3f}, y={sample['sampled_y']:.3f}, z={sample['sampled_z']:.3f}, yaw={sample['sampled_yaw']:.3f}")
        print(f"  Inverse:   x={inv_sample['sampled_x']:.3f}, y={inv_sample['sampled_y']:.3f}, z={inv_sample['sampled_z']:.3f}, yaw={inv_sample['sampled_yaw']:.3f}")


def verify_inverse_transform():
    # Test cases
    test_cases = [
        (1.0, 2.0, 0.3, np.pi/4),      # 45 degrees
        (0.5, -1.5, 0.2, np.pi/2),    # 90 degrees
        (-2.0, 1.0, 0.1, -np.pi/3),   # -60 degrees
        (0.0, 0.0, 0.0, 0.0),         # Zero
    ]
    
    for i, (x, y, z, yaw) in enumerate(test_cases):
        print(f"\nTest {i+1}: x={x}, y={y}, z={z}, yaw={yaw:.3f}")
        
        # Calculate inverse transform
        inv_x, inv_y, inv_z, inv_yaw = inverse_transform(x, y, z, yaw)
        print(f"  Inverse: x={inv_x:.3f}, y={inv_y:.3f}, z={inv_z:.3f}, yaw={inv_yaw:.3f}")
        
        # Inverse of inverse (should restore original values)
        back_x, back_y, back_z, back_yaw = inverse_transform(inv_x, inv_y, inv_z, inv_yaw)
        print(f"  Restored: x={back_x:.3f}, y={back_y:.3f}, z={back_z:.3f}, yaw={back_yaw:.3f}")
        
        # Check error
        error = np.sqrt((x - back_x)**2 + (y - back_y)**2 + (z - back_z)**2 + (yaw - back_yaw)**2)
        print(f"  Error: {error:.6f}")
        
        if error < 1e-10:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")


def main():
    parser = argparse.ArgumentParser(description="Perform inverse transform on sampled values in JSON file.")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-o", "--output", help="Output JSON file path (default: add '_inverse' to input_file)")
    parser.add_argument("--verify", action="store_true", help="Run inverse transform verification test")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_inverse_transform()
        return
    
    # Check input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file does not exist: {input_path}")
        return
    
    # Set output file path
    if args.output:
        output_path = Path(args.output)
    else:
        # Save to files/batch_results directory
        results_dir = Path("files/batch_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / (input_path.stem + "_inverse" + input_path.suffix)
    
    # Process inverse transform
    try:
        process_json_file(input_path, output_path)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return


if __name__ == "__main__":
    main() 