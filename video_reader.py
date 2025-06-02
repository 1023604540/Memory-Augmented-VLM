import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import yaml
from decord import VideoReader, cpu
import json
import multiprocessing

# Modify this for your own LLaVA dataset yaml path
DATA_YAML = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/test.yaml"
VIDEO_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos"
OUTPUT_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos_tensors"


# Create output directory if not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YAML file and parse full list_data_dict

# Pure video processing: 1 frame per second sampling
# def process_with_decord(video_file):
#     try:
#         vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
#         total_frame_num = len(vr)
#         avg_fps = vr.get_avg_fps()
#
#         avg_fps_round = max(1, round(avg_fps))  # 1 frame/sec
#         frame_idx = [i for i in range(0, total_frame_num, avg_fps_round)]
#
#         video = vr.get_batch(frame_idx).asnumpy()
#         return video
#
#     except Exception as e:
#         print(f"Failed to decode {video_file} with decord. Exception: {e}")
#         return None


def load_all_samples_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    all_samples = []
    for dataset in yaml_data["datasets"]:
        json_path = dataset["json_path"]
        with open(json_path, "r") as jf:
            samples = json.load(jf)
            all_samples.extend(samples)
    return all_samples

def process_one(item):
    video_file = item.get("video")
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    save_path = os.path.join(OUTPUT_FOLDER, video_file + ".pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # if os.path.exists(save_path):
    #     return

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()
        avg_fps_round = max(1, round(avg_fps))
        frame_idx = [i for i in range(0, total_frame_num, avg_fps_round)]
        video = vr.get_batch(frame_idx).asnumpy()
        torch.save(video, save_path)
        vr.seek(0)
    except Exception as e:
        print(f"Failed to decode {video_file}. Exception: {e}")

list_data_dict = load_all_samples_from_yaml(DATA_YAML)

with multiprocessing.Pool(processes=64) as pool:  # Adjust process count depending on your CPU cores
    list(tqdm(pool.imap(process_one, list_data_dict), total=len(list_data_dict)))
