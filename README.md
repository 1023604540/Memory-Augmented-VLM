<p align="center" width="100%">
</p>

# Memory Enhanced Video Language Model for Long Video Understanding


This project introduces a Memory-Augmented Video-Language Model built on top of the LLaVA-OneVision backbone. The model integrates a recurrent memory module with learnable memory tokens that evolve over time, consolidating both recent and historical visual context. A lightweight Memory-Fuser combines these representations with fine-grained frame embeddings, enabling scalable reasoning across extended temporal contexts.

We evaluate our approach on five long-video understanding benchmarks—LongVideoBench, VideoMME, MLVU, NExT-QA, and EgoSchema—achieving consistent improvements over the baseline. Unlike existing VLMs, our architecture maintains accuracy as video length increases, effectively bridging the gap between scalability and performance.


## Models & Scripts
The model installation follows exactly the LLaVA-OneVision setup. Please refer to the [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)
### Installation

#### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

#### 2. **Install the inference package:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

### Data Preparation
The training data come from [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) dataset. Please also refer to [LLaVA-OneVision Data Preparation](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/README.md) for data preparation instructions.

### Training
The training is conducted in two stages. The first stage is to train the memory module first on short videos and the second stage is to fine-tune the whole model on longer video range. The script for the first stage is in [scripts/train/finetune_short.sh]() and the script for the second stage is in [scripts/train/finetune_long.sh](). Please refer to the scripts for more details.

### Evaluaton
The evaluation is performed using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Please first install the lmms-eval package. To reproduce the results in our paper, you can use the following script:
```bash
accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \\
    --model llava_onevision \\
    --model_args pretrained={checkpoint_path},max_frames_num={frames},model_name={model_name},attn_implementation="flash_attention_2" \\
    --tasks longvideobench_val_v,videomme,mlvu_dev,nextqa_mc_test,egoschema \\
    --batch_size 1 \\
    --log_samples \\
    --log_samples_suffix {model_name}_frames{frames}_log \\
    --output_path ./logs_frame/
```

**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama-1/2 community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5, [Tongyi Qianwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/blob/main/LICENSE) and [Llama-3 Research License](https://llama.meta.com/llama3/license/)). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.
