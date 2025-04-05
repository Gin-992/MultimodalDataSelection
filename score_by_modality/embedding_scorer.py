import json
import os

from tqdm import tqdm

from score_by_modality.score_image_api import chat


def generate_rating(model, image_pth, question, options):
    image_fn = os.path.basename(image_pth)
    image_dir = os.path.dirname(image_pth)

    options = filter(lambda x: x != "nan", options)
    opt_str = ""
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        opt_str += f"{opt_str}{letter}. {option}\n"

    question = f"Question: {question}\n{opt_str}"

    url = "http://127.0.0.1:1234/v1/chat/completions"
    sys_msg = "You are a helpful assistant."
    prompt = f"""You are provided with an image and a question. Your task is to evaluate how well the image supports answering the question. Please follow these steps:

1. **Analyze the Image:** Carefully examine the image to identify relevant details such as visual cues, objects, text, and overall context.
2. **Understand the Question:** Read the question to determine what specific information is required.
3. **Evaluate Relevance:** Assess how effectively the image provides the necessary details to answer the question.
4. **Assign a Rating:** Based on your evaluation, assign a rating on a scale from 1 to 10, where:
   - 10 indicates that the image is extremely helpful in answering the question.
   - 1 indicates that the image provides little to no useful information.
5. **Output Format:** Provide only your final rating score in JSON format with a single key "rating". Do not include any additional text or explanation.

Example Output:
{{"rating": 8}}

Question: {question}
"""

    response = chat(model, sys_msg, prompt, image_fn, 2048, 0.9, url, image_dir)
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    with open("/mnt/shared_resources/datasets/Sample-MM-Data/sampled_bench.json", "r") as f:
        data = json.load(f)

    pass
    scores = []
    for entry in tqdm(data):
        question = entry["question"]
        options = entry["options"]

        gaussian_image = entry["image_paths"]["gaussian_noise"]
        salt_pepper_image = entry["image_paths"]["salt_pepper_noise"]
        untouched_image = entry["image_paths"]["untouched"]

        gaussian_image = os.path.join("/mnt/shared_resources/datasets/Sample-MM-Data/gaussian_noise", os.path.basename(gaussian_image))
        salt_pepper_image = os.path.join("/mnt/shared_resources/datasets/Sample-MM-Data/salt_pepper_noise",
                                      os.path.basename(salt_pepper_image))
        untouched_image = os.path.join("/mnt/shared_resources/datasets/Sample-MM-Data/untouched",
                                      os.path.basename(untouched_image))

        untouched_score = generate_rating("./Qwen2.5-VL-7B-Instruct", untouched_image, question, options)
        gaussian_score = generate_rating("./Qwen2.5-VL-7B-Instruct", gaussian_image, question, options)
        salt_pepper_score = generate_rating("./Qwen2.5-VL-7B-Instruct", salt_pepper_image, question, options)

        entry["untouched_score"] = untouched_score
        entry["gaussian_score"] = gaussian_score
        entry["salt_pepper_score"] = salt_pepper_score

        scores.append(entry)

    with open("/mnt/shared_resources/datasets/Sample-MM-Data/sampled_bench_emb_score.json", "w") as f:
        json.dump(scores, f, indent=4, default=str)
