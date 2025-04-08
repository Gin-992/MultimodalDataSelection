import os
import sys
import json
import argparse

from click import prompt
from tqdm import tqdm

# Import chat APIs from the corresponding modules.
from score_by_modality.score_image_api import chat as image_chat
from score_by_modality.score_text_api import chat as text_chat

CHAT_URL = "http://localhost:1234/v1/chat/completions"


def load_instructions(file_path):
    """Load the instruction set from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_results(data, output_file):
    """Save processed results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, default=str)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Task Processing Script")
    parser.add_argument("--task", type=str,
                        help="Task name to perform. Options: captioning, caption_scoring, embedding_scoring")
    parser.add_argument("--input_json", type=str, help="Path to the input JSON dataset")
    parser.add_argument("--output_json", type=str, help="Path to save the output JSON results")
    parser.add_argument("--instruction_file", type=str, default="instructions.json",
                        help="Path to the instruction set JSON file")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct",
                        help="Model name to use")
    return parser.parse_args()


# --- Chat Methods ---
def perform_image_task(model, image_path, instruction):
    """Perform an image-based task using the image API."""
    image_fn = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    url = "http://127.0.0.1:1234/v1/chat/completions"
    sys_msg = "You are a helpful assistant."
    response = image_chat(model, sys_msg, instruction, image_fn, 2048, 0.9, url, image_dir)
    return response['choices'][0]['message']['content']


def perform_caption_scoring(model, image_path, caption, instruction):
    """Score a caption (for a given image) using the text API."""
    sys_msg = "You are an evaluator for image captions."
    prompt = f"{instruction}\nEvaluate the following caption for the image {os.path.basename(image_path)}:\n{caption}"
    response = text_chat(model, sys_msg, prompt, caption, 2048, 0.9)
    return response['choices'][0]['message']['content']


def perform_embedding_scoring(model, text, instruction):
    """Score text embeddings using the text API."""
    sys_msg = "You are an evaluator for text embeddings."
    prompt = f"{instruction}\nEvaluate the following text:\n{text}"
    response = text_chat(model, sys_msg, prompt, text, 2048, 0.9)
    return response['choices'][0]['message']['content']


def perform_task_prediction(model, question, instruction):
    sys_msg = "You are a helpful assistant."
    prompt = f"{instruction}\nPrediction which task of the following question:\n{question}"
    response = text_chat(model, sys_msg, prompt, 256, 0.1, CHAT_URL)
    return response['choices'][0]['message']['content']


def perform_task_caption(model, task, image_path, instructions):
    image_fn = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)

    sys_msg = "You are a helpful assistant."
    task_mapping = {
        "Image Style": "image_style_captioning",
        "Image Scene": "image_scene_captioning",
        "Image Emotion": "image_emotion_captioning",
        "Image Quality": "image_quality_captioning",
        "Image Topic": "image_topic_captioning",
        "Object Localization": "object_localization_captioning",
        "Attribute Recognition": "attribute_recognition_captioning",
        "Celebrity Recognition": "celebrity_recognition_captioning",
        "OCR (Optical Character Recognition)": "ocr_captioning",
        "Spatial Relationship": "spatial_relation_captioning",
        "Attribute Comparison": "attribute_comparison_captioning",
        "Action Recognition": "action_recognition",
        "Physical Property Reasoning": "physical_property_captioning",
        "Function Reasoning": "function_captioning",
        "Identity Reasoning": "identity_captioning",
        "Social Relation": "social_relation_captioning",
        "Physical Relation": "physical_relation_captioning",
        "Nature Relation": "nature_relation_captioning",
        "Structuralized Image-Text Understanding": "structuralized_image_text_captioning",
        "Future Prediction": "future_prediction_captioning"
    }
    prompt = f"{instructions[task_mapping[task]]}"
    response = image_chat(model, sys_msg, prompt, image_fn, 2048, 0.1, CHAT_URL, image_dir)
    return response['choices'][0]['message']['content']


# --- Post-Processing Functions ---
def process_captioning(data, instruction, model):
    """
    Process image captioning:
    For each entry, generate captions for different image variations.
    """
    for entry in tqdm(data, desc="Captioning images"):
        entry["untouched_caption"] = perform_image_task(model, entry["image_paths"]["untouched"], instruction)
        entry["gaussian_caption"] = perform_image_task(model, entry["image_paths"]["gaussian_noise"], instruction)
        entry["salt_pepper_caption"] = perform_image_task(model, entry["image_paths"]["salt_pepper_noise"], instruction)
    return data


def process_caption_scoring(data, instruction, model):
    """
    Process caption scoring:
    For each entry, score existing captions for different image variations.
    """
    for entry in tqdm(data, desc="Scoring captions"):
        # It is assumed that captions are already generated in the entry.
        entry["untouched_caption_score"] = perform_caption_scoring(
            model, entry["image_paths"]["untouched"], entry.get("untouched_caption", ""), instruction)
        entry["gaussian_caption_score"] = perform_caption_scoring(
            model, entry["image_paths"]["gaussian_noise"], entry.get("gaussian_caption", ""), instruction)
        entry["salt_pepper_caption_score"] = perform_caption_scoring(
            model, entry["image_paths"]["salt_pepper_noise"], entry.get("salt_pepper_caption", ""), instruction)
    return data


def process_embedding_scoring(data, instruction, model):
    """
    Process embedding scoring:
    For each entry, score a text embedding.
    It is assumed that each entry contains a 'text_input' field.
    """
    for entry in tqdm(data, desc="Scoring embeddings"):
        entry["embedding_score"] = perform_embedding_scoring(model, entry.get("text_input", ""), instruction)
    return data


def process_task_prediction(data, instruction, model):
    for entry in tqdm(data, desc="Task prediction"):
        options = entry["options"]
        question = entry["question"]
        options = filter(lambda x: x != "nan", options)
        opt_str = ""
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            opt_str += f"{opt_str}{letter}. {option}\n"

        question = f"Question: {question}\n{opt_str}"
        entry["predicted_task"] = perform_task_prediction(model, question, instruction)
    return data

def process_task_captioning(data, instruction, model):
    for entry in tqdm(data, desc="Task captioning"):
        # image_path = os.path.join("/Volumes/Ming-Data/Dataset/MM-Data/untouched", os.path.basename(entry["image_paths"]["untouched"]))
        # entry["caption"] = perform_task_caption(model, entry["sub-task"], image_path, instruction)
        entry["caption"] = perform_task_caption(model, entry["sub-task"], entry["image_paths"]["untouched"], instruction)
    return data


# --- Task Processor Mapping ---
TASK_PROCESSORS = {
    "captioning": process_captioning,
    "score_caption_question": process_caption_scoring,
    "score_embedding_question": process_embedding_scoring,
    "predict_task": process_task_prediction,
    "task_captioning": process_task_captioning,
}


def get_instruction_for_task(task, instructions):
    """Retrieve the instruction prompt for a given task."""
    if task in instructions:
        return instructions[task]
    else:
        return None


def main():
    args = parse_args()

    # Load instructions from the instruction set file.
    instructions = load_instructions(args.instruction_file)
    instruction = get_instruction_for_task(args.task, instructions)
    if instruction is None:
        print(f"Task '{args.task}' not found in instruction file. Please check available tasks.")
        sys.exit(1)

    # Load input dataset.
    with open(args.input_json, "r") as f:
        data = json.load(f)

    # Choose the processor for the task.
    processor = TASK_PROCESSORS.get(args.task)
    if processor is None:
        print(f"No processor found for task '{args.task}'.")
        sys.exit(1)

    # Process data sequentially using the chosen chat and post-processing methods.
    processed_data = processor(data, instruction, args.model)

    # Save the processed results.
    save_results(processed_data, args.output_json)
    print(f"Processing completed. Results saved to {args.output_json}.")


if __name__ == '__main__':
    main()
