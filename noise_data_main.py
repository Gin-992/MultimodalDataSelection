import json
import os
from pathlib import Path
import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from protuberance.image.noise_factory import NoiseFactory

# 配置常量集中管理 [1,5](@ref)
CONFIG = {
    "HOME_DIR": Path("/Volumes/Ming-Data/Dataset/Sample-MM-Data"),
    "SOURCE_DATA_PATHS": {
        "cn": "/Volumes/Ming-Data/Dataset/MMBench/cn/*.parquet",
        "en": "/Volumes/Ming-Data/Dataset/MMBench/en/*.parquet",
        "cc": "/Volumes/Ming-Data/Dataset/MMBench/cc/*.parquet"
    },
    "NOISE_TYPES": ["gaussian", "salt_pepper"],
    "MAX_SAMPLES": 100
}


def create_directories(base_path: Path) -> dict:
    """创建所有需要的目录并返回路径字典 [6](@ref)"""
    dirs = {
        "gaussian_noise": base_path / "gaussian_noise",
        "salt_pepper_noise": base_path / "salt_pepper_noise",
        "untouched": base_path / "untouched"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def process_dataset_entry(index: int, entry: dict, output_dirs: dict) -> dict:
    """处理单个数据集条目，生成标准化记录 [3,7](@ref)"""
    return {
        "question": entry["question"],
        "options": [entry["A"], entry["B"], entry["C"], entry["D"]],
        "answer": entry["answer"],
        "metadata": {
            "hint": entry["hint"],
            "category": entry["category"],
            "l2_category": entry["L2-category"],
            "comment": entry["comment"],
            "split": entry["split"]
        },
        "image_paths": {
            "untouched": output_dirs["untouched"] / f"{index}.jpg",
            "gaussian_noise": output_dirs["gaussian_noise"] / f"{index}.jpg",
            "salt_pepper_noise": output_dirs["salt_pepper_noise"] / f"{index}.jpg"
        }
    }


def apply_image_processing(image, output_paths: dict):
    """统一处理图像保存逻辑 [4,8](@ref)"""
    try:
        np_image = np.array(image)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # 应用噪声处理
        cv2.imwrite(str(output_paths["gaussian_noise"]),
                    NoiseFactory.apply_noise(cv_image, "gaussian", sigma=30))
        cv2.imwrite(str(output_paths["salt_pepper_noise"]),
                    NoiseFactory.apply_noise(cv_image, "salt_pepper", prob=0.05))
        image.save(str(output_paths["untouched"]))
    except Exception as e:
        print(f"图像处理失败：{str(e)}")


def main():
    """主处理流程 [2,5](@ref)"""
    # 初始化目录
    output_dirs = create_directories(CONFIG["HOME_DIR"])

    # 加载数据集
    dataset = load_dataset("parquet", data_files=CONFIG["SOURCE_DATA_PATHS"])
    filtered_dataset = dataset["en"].filter(
        lambda ex: ex["category"] == "object_localization"
    )

    sampled_data = []
    for idx in tqdm(range(min(filtered_dataset.num_rows, CONFIG["MAX_SAMPLES"]))):
        entry = filtered_dataset[idx]

        # 跳过非目标类别数据
        if entry["category"] != "object_localization":
            continue

        # 生成记录
        record = process_dataset_entry(len(sampled_data) + 1, entry, output_dirs)
        apply_image_processing(entry["image"], record["image_paths"])

        sampled_data.append(record)
        if len(sampled_data) >= CONFIG["MAX_SAMPLES"]:
            break

    # 保存元数据
    with open(CONFIG["HOME_DIR"] / "sampled_bench.json", "w") as f:
        json.dump(sampled_data, f, indent=4, default=str)


if __name__ == '__main__':
    main()
