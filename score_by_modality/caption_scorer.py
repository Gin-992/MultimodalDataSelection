import json
import os

from tqdm import tqdm

from score_by_modality.score_text_api import chat


def generate_rating(model, caption, question, options):
    options = filter(lambda x: x != "nan", options)
    opt_str = ""
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        opt_str += f"{opt_str}{letter}. {option}\n"

    question = f"Question: {question}\n{opt_str}"

    url = "http://127.0.0.1:1234/v1/chat/completions"
    sys_msg = "You are a helpful assistant."
    prompt = f"""You are provided with a caption and a question. Your task is to evaluate how effectively the caption supplies the necessary details to answer the question. Please follow these steps:

1. **Read the Caption:** Carefully review the caption to understand the information it contains.
2. **Identify Key Details:** Extract the important details from the caption that might be relevant to the question.
3. **Review the Question:** Analyze the question to determine what specific information is required to answer it.
4. **Evaluate Relevance and Completeness:** Assess how well the caption covers the needed details for the question. Consider factors such as clarity, specificity, and completeness.
5. **Assign a Rating:** Based on your analysis, rate the helpfulness of the caption on a scale from 1 to 10, where:
   - 10 means the caption is fully comprehensive and directly addresses the question.
   - 1 means the caption provides little to no useful information for answering the question.
6. **Explain Your Rating:** Provide a clear and concise explanation for the rating, citing key aspects of the caption that support your decision.

Output Format:
{{"rating": 1, "explanation": "A brief explanation justifying your rating."}}

Caption: {caption}

Question: {question}
"""

    response = chat(model, sys_msg, prompt, 2048, 0.9, url)
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    with open("/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_bench_caption.json", "r") as f:
        data = json.load(f)

    pass
    scores = []
    for entry in tqdm(data):
        question = entry["question"]
        options = entry["options"]
        gaussian_caption = entry["gaussian_caption"]
        salt_pepper_caption = entry["salt_pepper_caption"]
        untouched_caption = entry["untouched_caption"]

        untouched_score = generate_rating("./Qwen2.5-VL-7B-Instruct", untouched_caption, question, options)
        gaussian_score = generate_rating("./Qwen2.5-VL-7B-Instruct", gaussian_caption, question, options)
        salt_pepper_score = generate_rating("./Qwen2.5-VL-7B-Instruct", salt_pepper_caption, question, options)

        entry["untouched_caption"] = untouched_caption
        entry["gaussian_caption"] = gaussian_caption
        entry["salt_pepper_caption"] = salt_pepper_caption

        scores.append(entry)

    with open("/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_bench_scores.json", "w") as f:
        json.dump(scores, f, indent=4, default=str)
