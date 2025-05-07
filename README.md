# Multimodal Data Evaluation Pipeline

## 📦 Requirements

- **Dataset**  
  ```bash
  git clone https://www.modelscope.cn/datasets/Endlinc/MMDS-SampledDataPool.git
  ```

**Model Weights**

  ```bash
  # Base training models
  git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
  git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

  # Rating models
  git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ
  git clone https://huggingface.co/zhangzicheng/q-sit
  git clone https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ
  ```

## ⚙️ Setup

1. Create and activate a virtual environment:

   ```bash
   conda create -n mm-data python=3.11 -y
   conda activate mm-data
   pip install -r pipeline-requirements.txt
   ```
2. Install and configure vLLM for GPU inference.

## 🚀 Usage

```bash
# Concat all image file tars and untar
python launch_job.py --job_config untar_images.yaml

# (Optional) Split annotation file for multi-GPU inference
python launch_job.py --job_config split_json.yaml

# Task prediction
python launch_job.py --job_config gen_task_prediction.yaml

# Parse predictions
python launch_job.py --job_config retrieve_task_domin.yaml

# Task-specific captioning
python launch_job.py --job_config gen_task_cap.yaml

# General captioning (DONE)
python launch_job.py --job_config gen_g_cap.yaml

# Multimodal rating
python launch_job.py --job_config gen_mm_score.yaml

# Parse MM rating
python launch_job.py --job_config retrieve_mm_score_domin.yaml

# Image quality rating
python launch_job.py --job_config gen_image_quality.yaml

# Text quality rating
python launch_job.py --job_config gen_text_quality.yaml

# Parse Text rating
python launch_job.py --job_config retrieve_text_score_domin.yaml
```

Each `--job_config` flag points to a YAML file in the `data_eval_scripts/` and `helper_scripts/` directory that specifies:

## 📄 Output

* **Task predictions:** `outputs/task_predictions.jsonl`
* **Captions:**

  * Task-specific: `outputs/task_captions.jsonl`
  * General: `outputs/general_captions.jsonl`
* **Scores:**

  * Multimodal: `outputs/mm_scores.csv`
  * Image quality: `outputs/image_quality.csv`
  * Text quality: `outputs/text_quality.csv`
