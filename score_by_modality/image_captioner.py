import json
import os

from tqdm import tqdm

from score_by_modality.score_image_api import chat

def generate_caption(model, image_pth):
    image_fn = os.path.basename(image_pth)
    image_dir = os.path.dirname(image_pth)

    url = "http://127.0.0.1:1234/v1/chat/completions"
    sys_msg = "You are a helpful assistant."
    prompt = """Please carefully observe the image below and provide a detailed description based on what you see. Please include, but do not limit yourself to, the following aspects:
1. **Overall Overview**: Describe the main scene, environment, and background atmosphere of the image.
2. **Subject and Details**: Identify the primary person or object in the image, detailing their appearance, posture, expression, and any specific features such as clothing.
3. **Color and Light**: Analyze the image’s color tone, lighting effects, contrast, and color palette, and explain how these elements influence the overall mood.
4. **Composition and Artistic Style**: Discuss the image’s composition, perspective, and any artistic style that might be present (e.g., realism, abstraction, etc.).
5. **Implied Information and Emotions**: Speculate on the emotions, story, or symbolism the image may convey, and consider its possible cultural background or artistic intention.

Please ensure that your description is thorough, vivid, and professional, covering as many details as possible in the image.
"""

    response = chat(model, sys_msg, prompt, image_fn, 2048, 0.9, url, image_dir)
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    with open("/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_bench.json", "r") as f:
        data = json.load(f)

    pass
    caption = []
    for entry in tqdm(data):
        gaussian_image = entry["image_paths"]["gaussian_noise"]
        salt_pepper_image = entry["image_paths"]["salt_pepper_noise"]
        untouched_image = entry["image_paths"]["untouched"]

        untouched_caption = generate_caption("qwen2.5-vl-7b-instruct", untouched_image)
        gaussian_caption = generate_caption("qwen2.5-vl-7b-instruct", gaussian_image)
        salt_pepper_caption = generate_caption("qwen2.5-vl-7b-instruct", salt_pepper_image)

        entry["untouched_caption"] = untouched_caption
        entry["gaussian_caption"] = gaussian_caption
        entry["salt_pepper_caption"] = salt_pepper_caption

        caption.append(entry)

    with open("/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_bench_caption.json", "w") as f:
        json.dump(caption, f, indent=4, default=str)
