import os
import yaml  # 使用 PyYAML 解析 YAML

def check_yaml_paths(config_path):
    # 读取 YAML 文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # 解析 YAML
    
    missing_files = []

    # 遍历 datasets
    for dataset in config.get("datasets", []):
        json_path = dataset.get("json_path")
        if json_path and not os.path.exists(json_path):
            missing_files.append(json_path)

    if missing_files:
        print("Missing JSON files:")
        for path in missing_files:
            print(path)
    else:
        print("All JSON files exist.")

# 替换成你的 YAML 配置文件路径
config_file_path = "/home/vault/b232dd/b232dd21/vlm/LLaVA-NeXT/scripts/train/onevision.yaml"
check_yaml_paths(config_file_path)
