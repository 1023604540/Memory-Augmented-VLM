import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import yaml
from decord import VideoReader, cpu

# Modify this for your own LLaVA dataset yaml path
DATA_YAML = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/test.yaml"
VIDEO_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos"
OUTPUT_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos_tensors"
FRAMES_UPBOUND = 300
IMG_SIZE = 384

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load your list_data_dict from YAML
with open(DATA_YAML, "r") as f:
    list_data_dict = yaml.safe_load(f)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def process_shareVideoGPTV(video_file, frames_upbound):
    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
    frame_files.sort()

    total_frames = len(frame_files)
    num_frames_to_sample = min(frames_upbound, total_frames)
    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    frames = []
    for idx in sampled_indices:
        frame_path = frame_files[idx]
        try:
            with Image.open(frame_path) as img:
                frame = img.convert("RGB")
                frame_tensor = transform(frame)
                frames.append(frame_tensor)
        except IOError:
            print(f"Failed to read frame at path: {frame_path}")
    return torch.stack(frames)

def process_with_decord(video_file):
    try:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        # Sample 1 frame per second
        avg_fps_round = max(1, round(avg_fps))
        frame_idx = [i for i in range(0, total_frame_num, avg_fps_round)]

        video = vr.get_batch(frame_idx).asnumpy()

        frames = []
        for frame_np in video:
            frame = Image.fromarray(frame_np).convert("RGB")
            frame_tensor = transform(frame)
            frames.append(frame_tensor)

        vr.seek(0)
        return torch.stack(frames)

    except Exception as e:
        print(f"Failed to decode {video_file} with decord. Exception: {e}")
        return None

for item in tqdm(list_data_dict):
    video_file = item.get("video")
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    save_path = os.path.join(OUTPUT_FOLDER, video_file + ".pt")

    if os.path.exists(save_path):
        continue  # already cached

    if "shareVideoGPTV" in video_path:
        tensor = process_shareVideoGPTV(video_path, FRAMES_UPBOUND)
    else:
        tensor = process_with_decord(video_path)
        if tensor is None:
            continue

    torch.save(tensor, save_path)
