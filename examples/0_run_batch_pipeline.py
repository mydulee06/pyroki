#!/usr/bin/env python3
"""
Batch Processing Pipeline
Runs the complete pipeline: batch_eetrack -> inverse_transform -> json_to_pt_converter -> cma_es

Usage:
python run_batch_pipeline.py --sit_target_height 0.37
python run_batch_pipeline.py --sit_target_height 0.35 --skip_batch_eetrack
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import time

def run_command(cmd, description, check=True, capture_output=False):
    """Run a command with error handling"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result
        else:
            result = subprocess.run(cmd, check=check)
            
        elapsed = time.time() - start_time
        print(f"✅ {description} completed successfully in {elapsed:.1f}s")
        return result
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f}s")
        print(f"Error code: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise

def check_file_exists(filepath, description):
    """Check if file exists and print info"""
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"✅ {description} exists: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description} not found: {filepath}")
        return False

def count_successful_samples(json_file):
    """Count successful samples in JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            total = len(data)
            successful = sum(1 for item in data if item.get('success', False))
            success_rate = successful / total * 100 if total > 0 else 0
            return successful, total, success_rate
        else:
            return 0, 0, 0
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description='Run complete batch processing pipeline')
    parser.add_argument('--sit_target_height', type=float, default=0.37,
                       help='Target height for sitting position in meters (default: 0.37)')
    parser.add_argument('--skip_batch_eetrack', action='store_true',
                       help='Skip batch_eetrack step (use existing results)')
    parser.add_argument('--skip_inverse_transform', action='store_true',
                       help='Skip inverse_transform step (use existing results)')
    parser.add_argument('--skip_json_to_pt', action='store_true',
                       help='Skip json_to_pt_converter step (use existing results)')
    parser.add_argument('--skip_cma_es', action='store_true',
                       help='Skip cma_es step')
    parser.add_argument('--exp_prefix', type=str, default=None,
                       help='Experiment prefix for CMA-ES (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Setup paths
    height_cm = int(args.sit_target_height * 100)
    results_dir = Path("files/batch_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    batch_results_file = results_dir / f"batch_eetrack_results_{height_cm}.json"
    inverse_results_file = results_dir / f"batch_eetrack_results_{height_cm}_inverse.json"
    pt_file = results_dir / f"batch_eetrack_results_{height_cm}_inverse.pt"
    
    # Set experiment prefix if not provided
    if args.exp_prefix is None:
        args.exp_prefix = f"batch_pipeline_h{height_cm}"
    
    print(f"🎯 Batch Processing Pipeline")
    print(f"Sit target height: {args.sit_target_height}m ({height_cm}cm)")
    print(f"Experiment prefix: {args.exp_prefix}")
    print(f"Results directory: {results_dir}")
    
    # Step 1: Run batch_eetrack
    if not args.skip_batch_eetrack:
        cmd = [
            sys.executable, "14_batch_eetrack.py",
            "--sit_target_height", str(args.sit_target_height)
        ]
        run_command(cmd, "Running batch EETrack optimization")
    else:
        print(f"\n⏭️  Skipping batch_eetrack step")
    
    # Check batch results
    if not check_file_exists(batch_results_file, "Batch EETrack results"):
        print("❌ Cannot proceed without batch results file")
        return 1
    
    # Show batch results summary
    successful, total, success_rate = count_successful_samples(batch_results_file)
    print(f"📊 Batch results: {successful}/{total} successful ({success_rate:.1f}%)")
    
    # Step 2: Run inverse_transform
    if not args.skip_inverse_transform:
        cmd = [
            sys.executable, "utils/inverse_transform.py",
            str(batch_results_file)
        ]
        run_command(cmd, "Running inverse transform")
    else:
        print(f"\n⏭️  Skipping inverse_transform step")
    
    # Check inverse results
    if not check_file_exists(inverse_results_file, "Inverse transform results"):
        print("❌ Cannot proceed without inverse results file")
        return 1
    
    # Show inverse results summary
    inv_successful, inv_total, inv_success_rate = count_successful_samples(inverse_results_file)
    print(f"📊 Inverse results: {inv_successful}/{inv_total} successful ({inv_success_rate:.1f}%)")
    
    # Step 3: Run json_to_pt_converter
    if not args.skip_json_to_pt:
        cmd = [
            sys.executable, "utils/json_to_pt_converter.py",
            "--input", str(inverse_results_file),
            "--output", str(pt_file)
        ]
        run_command(cmd, "Converting JSON to PT format")
    else:
        print(f"\n⏭️  Skipping json_to_pt_converter step")
    
    # Check PT file
    if not check_file_exists(pt_file, "PT format file"):
        print("❌ Cannot proceed without PT file")
        return 1
    
    # Step 4: Prepare CMA-ES directory structure and run CMA-ES
    if not args.skip_cma_es:
        # Create CMA-ES expected directory structure
        cma_exp_dir = Path("files") / args.exp_prefix
        cma_subdir = cma_exp_dir / "dummy_exp"  # CMA-ES expects subdirectories
        cma_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy PT file to expected location
        cma_pt_file = cma_subdir / "batch_eetrack_results_inverse.pt"
        import shutil
        shutil.copy2(pt_file, cma_pt_file)
        print(f"📂 Copied PT file to CMA-ES expected location: {cma_pt_file}")
        
        cmd = [
            sys.executable, "utils/cma_es.py",
            "--log_dir", "files",
            "--exp_prefix", args.exp_prefix,
            "--algo", "cmaes",
            "--save_cmaes_result",
            "--animate"
        ]
        run_command(cmd, "Running CMA-ES optimization")
    else:
        print(f"\n⏭️  Skipping cma_es step")
    
    # Final summary
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Results directory: {results_dir}")
    print(f"📄 Files generated:")
    print(f"   - Batch results: {batch_results_file}")
    print(f"   - Inverse results: {inverse_results_file}")
    print(f"   - PT file: {pt_file}")
    print(f"   - CMA-ES results: files/{args.exp_prefix}/")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        sys.exit(1) 