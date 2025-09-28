import torch
import numpy as np
import argparse

LAB_JOINT = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint',
]

parser = argparse.ArgumentParser(description="Convert sit_terminal_states pt file to npz file.")
parser.add_argument('sit_terminal_states_path', type=str, help='Path to the sit terminal states path (Ex: eetrack/terminal_states_11538.pt)')
args = parser.parse_args()

data = torch.load(args.sit_terminal_states_path, weights_only=False)
data = {k: v.cpu().numpy() for k, v in data.items()}
data["lab_joint"] = LAB_JOINT

save_path = args.sit_terminal_states_path.replace(".pt", ".npz")
np.savez(save_path, **data)

print(f"npz file saved to {save_path}.")
