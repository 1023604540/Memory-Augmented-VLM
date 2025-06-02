import os
import numpy as np
import torch
from tqdm import tqdm
import yaml
from decord import VideoReader, cpu
import json
import multiprocessing
import time

# =================== CONFIGURATION =====================
DATA_YAML = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/test2-3.yaml"
VIDEO_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos"
OUTPUT_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos_tensors"
PROCESS_COUNT = 64     # Try 8, 16, 32, 64 -- tune for your node!
DECOD_THREADS = 1       # Tune per process; 1 is usually best unless you use fewer processes
SKIP_EXISTING = True    # Skip videos that are already processed
ERROR_LOG = os.path.join(OUTPUT_FOLDER, "error_log.txt")
# =======================================================

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

def deduplicate(items):
    seen = set()
    unique_items = []
    for item in items:
        v = item.get("video")
        if v not in seen:
            unique_items.append(item)
            seen.add(v)
    return unique_items

def process_one(item):
    video_file = item.get("video")
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    save_path = os.path.join(OUTPUT_FOLDER, video_file + ".pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if SKIP_EXISTING and os.path.exists(save_path):
        print(f"Skipping {video_file}, already exists.", flush=True)
        return

    try:
        start = time.time()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=DECOD_THREADS)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()
        avg_fps_round = max(1, round(avg_fps))
        frame_idx = [i for i in range(0, total_frame_num, avg_fps_round)]
        video = vr.get_batch(frame_idx).asnumpy()
        torch.save(video, save_path)
        print(f"Processed {video_file} in PID {os.getpid()} in {time.time()-start:.2f}s", flush=True)
    except Exception as e:
        with open(ERROR_LOG, "a") as ef:
            ef.write(f"Failed to decode {video_file}. Exception: {e}\n")
        print(f"Failed to decode {video_file}. Exception: {e}", flush=True)

if __name__ == "__main__":
    # Load and deduplicate video list
    print(f"Loading video list from {DATA_YAML}", flush=True)
    all_samples = load_all_samples_from_yaml(DATA_YAML)
    unique_samples = deduplicate(all_samples)
    print(f"Loaded {len(all_samples)} entries, deduplicated to {len(unique_samples)} unique videos.", flush=True)
    print(f"Using {PROCESS_COUNT} processes and {DECOD_THREADS} decord threads per process.", flush=True)

    with multiprocessing.Pool(processes=PROCESS_COUNT) as pool:
        list(tqdm(pool.imap(process_one, unique_samples), total=len(unique_samples)))
