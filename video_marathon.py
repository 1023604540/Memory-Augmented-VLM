import pandas as pd
import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load parquet file
df = pd.read_parquet('/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videomarathon/videomarathon/data/train-00000-of-00001.parquet')

# Ensure video folder exists
video_folder = '/hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videomarathon/videomarathon/videos'
os.makedirs(video_folder, exist_ok=True)

# Download function
def download_video(row):
    relative_path = row['video']
    url = row['URL']
    filename = os.path.join(video_folder, relative_path.split('/')[-1])

    if not os.path.exists(filename):  # Avoid re-downloading
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return row  # return row unchanged for further processing

# Parallel downloading
max_workers = 16  # Adjust this according to your SLURM node CPU count

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_video, row) for _, row in df.iterrows()]

    # Wrap futures in tqdm for progress bar
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

# After downloads are complete, build the dataset
mc_list = []
oe_list = []

for _, row in df.iterrows():
    qa_type = row['question_type'].split('/')[-1]

    entry = {
        "id": row['id'],
        "conversations": [
            {"from": "human", "value": f"<image>\n{row['question']}"},
            {"from": "gpt", "value": row['answer']}
        ],
        "data_source": row['data_source'],
        "video": row['video']
    }

    if qa_type == 'mc':
        mc_list.append(entry)
    elif qa_type == 'oe':
        oe_list.append(entry)

# Save JSON
with open('mc_questions.json', 'w') as f:
    json.dump(mc_list, f, indent=4)

with open('oe_questions.json', 'w') as f:
    json.dump(oe_list, f, indent=4)

print("Finished processing.")
