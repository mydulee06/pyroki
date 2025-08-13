"""
run_batch_pipeline_sweep.py
지정한 z_height에 대해 sit_target_height를 0.3~0.58 (0.02 간격)로 sweep하며 0_run_batch_pipeline.py를 반복 실행

사용 예시:
    python run_batch_pipeline_sweep.py --z_height 0.1
"""
import subprocess
import sys
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sweep sit_target_height for a given z_height and run 0_run_batch_pipeline.py repeatedly.")
    parser.add_argument('--z_height', type=float, required=True, help='z_height 값 (예: 0.1)')
    parser.add_argument('--quiet', action='store_true', help='0_run_batch_pipeline.py 실행 시 --quiet 옵션 추가')
    parser.add_argument('--skip_cma_es', action='store_true', help='CMA-ES 단계 건너뛰기')
    args = parser.parse_args()

    sit_target_heights = np.arange(0.42, 0.58 + 1e-6, 0.02)
    for sit_target_height in sit_target_heights:
        cmd = [
            sys.executable, '0_run_batch_pipeline.py',
            '--sit_target_height', f'{sit_target_height:.2f}',
            '--z_height', str(args.z_height)
        ]
        if args.quiet:
            cmd.append('--quiet')
        if args.skip_cma_es:
            cmd.append('--skip_cma_es')
        print(f"\n=== Running: sit_target_height={sit_target_height:.2f}, z_height={args.z_height} ===")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed for sit_target_height={sit_target_height:.2f}: {e}")
            continue

if __name__ == "__main__":
    main()
