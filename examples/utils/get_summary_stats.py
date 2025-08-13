"""
summary_dataset.json에서 z_height, sit_target_height로 x, y, yaw, dx, dy, dyaw 값을 조회하는 함수
사용 예시:
    from get_summary_stats import get_summary_stats
    x, y, yaw, dx, dy, dyaw = get_summary_stats(z_height=0.1, sit_target_height=0.37)
"""
import json
from pathlib import Path
import numpy as np

def get_summary_stats(z_height, sit_target_height, dataset_path=None):
    """
    summary_dataset.json에서 해당 (z_height, sit_target_height)에 해당하는 x, y, yaw, dx, dy, dyaw 값을 반환
    Args:
        z_height (float): z_height 값 (예: 0.1)
        sit_target_height (float): sit_target_height 값 (예: 0.37)
        dataset_path (str or Path, optional): summary_dataset.json 경로 (기본값: files/batch_results/summary_dataset.json)
    Returns:
        (x, y, yaw, dx, dy, dyaw) tuple
    Raises:
        ValueError: 해당 조합이 없을 때
    """
    if dataset_path is None:
        dataset_path = Path('files/batch_results/summary_dataset.json')
    else:
        dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    # float 비교는 isclose로
    for row in dataset:
        if np.isclose(row['z_height'], z_height) and np.isclose(row['target_height'], sit_target_height):
            return (
                row['x'], row['y'], row['yaw'],
                row['dx'], row['dy'], row['dyaw']
            )
    raise ValueError(f"No entry found for z_height={z_height}, sit_target_height={sit_target_height}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="summary_dataset.json에서 값 조회")
    parser.add_argument('--z_height', type=float, required=True)
    parser.add_argument('--sit_target_height', type=float, required=True)
    args = parser.parse_args()
    try:
        x, y, yaw, dx, dy, dyaw = get_summary_stats(args.z_height, args.sit_target_height)
        print(f"x={x}, y={y}, yaw={yaw}, dx={dx}, dy={dy}, dyaw={dyaw}")
    except Exception as e:
        print(f"Error: {e}")
