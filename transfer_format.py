# transfer Genie Sim data to LeRobot dataset format
import os
import h5py
import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import decord

decord.bridge.set_bridge('torch')

BASE_DIR = "/home/csl/genie-sim/genie_sim-origin/source/data_collection/recording_data/[sort_the_fruit_into_the_box_apple_g2_5]"
H5_PATH = os.path.join(BASE_DIR, "aligned_joints_all.h5")
VIDEO_DIR = os.path.join(BASE_DIR, "observations/videos")
OUTPUT_DIR = "/home/csl/genie-sim/genie_sim-origin/source/data_collection/outputs/lerobot_datasets/s1_fruit_bimanual"

print("---reading Genie Sim data---")
with h5py.File(H5_PATH, "r") as f:
    joint_states = f["state/joint/position"][:]
    joint_actions = f["action/joint/position"][:]
num_frames = joint_states.shape[0]  # 452
num_joints = joint_states.shape[1]  # 46

features_cfg = {
    "action": {"dtype": "float32", "shape": [num_joints], "names": [f"joint_{i}" for i in range(num_joints)]},
    "observation.state": {"dtype": "float32", "shape": [num_joints]},
   
    "observation.images.head": {"dtype": "video", "shape": [3, 400, 640], "names": ["head_rgb"]},
    "observation.images.left_hand": {"dtype": "video", "shape": [3, 1056, 1280], "names": ["left_hand_rgb"]},
    "observation.images.right_hand": {"dtype": "video", "shape": [3, 1056, 1280], "names": ["right_hand_rgb"]},
   
    "observation.depth_images.head": {"dtype": "video", "shape": [3, 400, 640], "names": ["head_depth"]},
    "observation.depth_images.left_hand": {"dtype": "video", "shape": [3, 1056, 1280], "names": ["left_hand_depth"]},
    "observation.depth_images.right_hand": {"dtype": "video", "shape": [3, 1056, 1280], "names": ["right_hand_depth"]},
}

dataset = LeRobotDataset.create(
    repo_id=OUTPUT_DIR,
    fps=30,
    features=features_cfg,
)

print("---reading videos with decord---")
vr_head_rgb    = decord.VideoReader(os.path.join(VIDEO_DIR, "head_color.mp4"))
vr_left_rgb    = decord.VideoReader(os.path.join(VIDEO_DIR, "hand_left_color.mp4"))
vr_right_rgb   = decord.VideoReader(os.path.join(VIDEO_DIR, "hand_right_color.mp4"))

vr_head_depth  = decord.VideoReader(os.path.join(VIDEO_DIR, "head_depth.mp4"))
vr_left_depth  = decord.VideoReader(os.path.join(VIDEO_DIR, "hand_left_depth.mp4"))
vr_right_depth = decord.VideoReader(os.path.join(VIDEO_DIR, "hand_right_depth.mp4"))

print("---LeRobot dataset loading---")
for t in range(num_frames):
    idx = min(t, num_frames - 1)
   
    frame_head_rgb  = vr_head_rgb[min(t, len(vr_head_rgb)-1)].byte()
    frame_left_rgb  = vr_left_rgb[min(t, len(vr_left_rgb)-1)].byte()
    frame_right_rgb = vr_right_rgb[min(t, len(vr_right_rgb)-1)].byte()

    frame_head_depth  = vr_head_depth[min(t, len(vr_head_depth)-1)].byte()
    frame_left_depth  = vr_left_depth[min(t, len(vr_left_depth)-1)].byte()
    frame_right_depth = vr_right_depth[min(t, len(vr_right_depth)-1)].byte()

    img_head_rgb  = frame_head_rgb.permute(2, 0, 1)
    img_left_rgb  = frame_left_rgb.permute(2, 0, 1)
    img_right_rgb = frame_right_rgb.permute(2, 0, 1)

    img_head_depth  = frame_head_depth[:, :, 0:1].permute(2, 0, 1).repeat(3, 1, 1)
    img_left_depth  = frame_left_depth[:, :, 0:1].permute(2, 0, 1).repeat(3, 1, 1)
    img_right_depth = frame_right_depth[:, :, 0:1].permute(2, 0, 1).repeat(3, 1, 1)

    frame_data = {
        "action": torch.tensor(joint_actions[idx]).float(),
        "observation.state": torch.tensor(joint_states[idx]).float(),
        "task": "sort the fruit into the box",  
       
        # 彩色影像 (uint8)
        "observation.images.head": img_head_rgb,
        "observation.images.left_hand": img_left_rgb,
        "observation.images.right_hand": img_right_rgb,
       
        # 深度影像 (uint8, 3通道)
        "observation.depth_images.head": img_head_depth,
        "observation.depth_images.left_hand": img_left_depth,
        "observation.depth_images.right_hand": img_right_depth,
    }
    dataset.add_frame(frame_data)

print("--- Parquet ---")

dataset.save_episode()

if hasattr(dataset, "consolidate"):
    dataset.consolidate()
elif hasattr(dataset.writer, "consolidate"):
    dataset.writer.consolidate()

print(f"saved to: {OUTPUT_DIR}")