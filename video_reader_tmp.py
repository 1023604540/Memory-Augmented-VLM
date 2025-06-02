import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
import yaml
from decord import VideoReader, cpu
import json
import multiprocessing
import time

# =================== CONFIGURATION =====================
DATA_YAML = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/tmp.yaml"
SHARED_VIDEO_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos"
SHARED_OUTPUT_FOLDER = "/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos_tensors"
PROCESS_COUNT = 8
DECOD_THREADS = 1
SKIP_EXISTING = True
# TMPDIR is set by the job system; fall back to /tmp
TMPDIR = os.environ.get("TMPDIR", "/tmp")
LOCAL_VIDEO_FOLDER = os.path.join(TMPDIR, "videos")
LOCAL_OUTPUT_FOLDER = os.path.join(TMPDIR, "videos_tensors")
ERROR_LOG = os.path.join(LOCAL_OUTPUT_FOLDER, "error_log.txt")
# =======================================================

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

def stage_in(videos_to_copy):
    print(f"Staging in {len(videos_to_copy)} videos to {LOCAL_VIDEO_FOLDER}...", flush=True)
    os.makedirs(LOCAL_VIDEO_FOLDER, exist_ok=True)
    for video_file in tqdm(videos_to_copy, desc="Copying videos"):
        src = os.path.join(SHARED_VIDEO_FOLDER, video_file)
        dst = os.path.join(LOCAL_VIDEO_FOLDER, video_file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):  # Only copy if not already there
            shutil.copy2(src, dst)
    print("Stage-in complete.", flush=True)

def process_one(item):
    video_file = item.get("video")
    video_path = os.path.join(LOCAL_VIDEO_FOLDER, video_file)
    save_path = os.path.join(LOCAL_OUTPUT_FOLDER, video_file + ".pt")
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

def stage_out(videos_to_copy):
    print(f"Staging out .pt files to {SHARED_OUTPUT_FOLDER}...", flush=True)
    for video_file in tqdm(videos_to_copy, desc="Copying tensors"):
        local_pt = os.path.join(LOCAL_OUTPUT_FOLDER, video_file + ".pt")
        shared_pt = os.path.join(SHARED_OUTPUT_FOLDER, video_file + ".pt")
        if os.path.exists(local_pt):
            os.makedirs(os.path.dirname(shared_pt), exist_ok=True)
            shutil.copy2(local_pt, shared_pt)
    print("Stage-out complete.", flush=True)

if __name__ == "__main__":
    print(f"Loading video list from {DATA_YAML}", flush=True)
    all_samples = load_all_samples_from_yaml(DATA_YAML)
    unique_samples = deduplicate(all_samples)
    video_file_list = [item['video'] for item in unique_samples]

    print(f"Loaded {len(all_samples)} entries, deduplicated to {len(unique_samples)} unique videos.", flush=True)
    print(f"Using {PROCESS_COUNT} processes and {DECOD_THREADS} decord threads per process.", flush=True)

    # --- 1. Stage in: copy videos to local scratch ---
    stage_in(video_file_list)

    # --- 2. Process videos using local input/output ---
    os.makedirs(LOCAL_OUTPUT_FOLDER, exist_ok=True)
    with multiprocessing.Pool(processes=PROCESS_COUNT) as pool:
        list(tqdm(pool.imap(process_one, unique_samples), total=len(unique_samples)))

    # --- 3. Stage out: copy results back to shared output folder ---
    stage_out(video_file_list)
