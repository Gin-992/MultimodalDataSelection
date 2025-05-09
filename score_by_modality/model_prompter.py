import os
import sys
import json
import argparse

from tqdm import tqdm

# Import chat APIs from the corresponding modules.
from score_by_modality.score_image_api import chat as image_chat
from score_by_modality.score_text_api import chat as text_chat

CHAT_URL = "http://localhost:1234/v1/chat/completions"
IMAGE_BASE = ""


# Helper function to concatenate conversations
def concat_conversations(conversations):
    """
    Concatenate a list of conversation dicts into a single string.
    Each dict has 'from' and 'value' keys.
    """
    # Format each message as "speaker: message"
    return "\n".join(f"{msg['from']}: {msg['value']}" for msg in conversations)


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
    parser.add_argument(
        "--image-base",
        type=str,
        default="",
        help="Base directory path for image files"
    )
    parser.add_argument("--input_json", type=str, help="Path to the input JSON dataset")
    parser.add_argument("--output_json", type=str, help="Path to save the output JSON results")
    parser.add_argument("--instruction_file", type=str, default="instructions.json",
                        help="Path to the instruction set JSON file")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct",
                        help="Model name to use")
    parser.add_argument(
        "--chat-url",
        type=str,
        default="http://localhost:1234/v1/chat/completions",
        help="URL for the chat completions API"
    )
    return parser.parse_args()


# --- Chat Methods ---
def perform_image_caption(model, image_path, instruction):
    """Perform an image-based task using the image API."""
    # Prepend IMAGE_BASE to the image path
    full_path = os.path.join(IMAGE_BASE, image_path)
    image_fn = os.path.basename(full_path)
    image_dir = os.path.dirname(full_path)
    sys_msg = "You are a helpful assistant."
    try:
        response = image_chat(model, sys_msg, instruction, image_fn, 2048, 0.9, CHAT_URL, image_dir)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Caption failed for {image_path!r}: {e}")
        return ""


def perform_caption_scoring(model, question, caption, instruction):
    """Score a caption (for a given image) using the text API."""
    sys_msg = "You are an evaluator for image captions."
    prompt = f"{instruction}\nCaption: {caption}\nQuestion: {question}"
    try:
        response = text_chat(model, sys_msg, prompt, 2048, 0.9, CHAT_URL)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Scoring failed for {caption!r}: {e}")
        return ""


def perform_task_prediction(model, question, instruction):
    sys_msg = "You are a helpful assistant."
    prompt = f"{instruction}\nPrediction which task of the following question:\n{question}"
    try:
        response = text_chat(model, sys_msg, prompt, 256, 0.1, CHAT_URL)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Prediction failed for {question!r}: {e}")
        return ""


def perform_task_caption(model, task, image_path, instructions):
    # Prepend IMAGE_BASE to the image path
    full_path = os.path.join(IMAGE_BASE, image_path)
    image_fn = os.path.basename(full_path)
    image_dir = os.path.dirname(full_path)

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
    try:
        response = image_chat(model, sys_msg, prompt, image_fn, 2048, 0.1, CHAT_URL, image_dir)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Caption failed for {image_path!r}: {e}")
        return ""


def perform_image_quality(model, image_path, instruction):
    """
    Perform an image quality assessment using the image API.
    """
    # Prepend IMAGE_BASE to the image path
    full_path = os.path.join(IMAGE_BASE, image_path)
    image_fn = os.path.basename(full_path)
    image_dir = os.path.dirname(full_path)
    sys_msg = "You are an evaluator for image quality assessment."
    prompt = instruction
    # Call the image API to assess quality
    try:
        response = image_chat(model, sys_msg, prompt, image_fn, 2048, 0.5, CHAT_URL, image_dir)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Quality assessment failed for {image_path!r}: {e}")
        return ""


def perform_text_quality(model, text, instruction):
    """
    Perform a text quality assessment using the text API.
    """
    sys_msg = "You are an evaluator for text quality assessment."
    prompt = f"{instruction}\nText: {text}"
    try:
        response = text_chat(model, sys_msg, prompt, 2048, 0.5, CHAT_URL)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Quality assessment failed for {text!r}: {e}")
        return ""


# --- Post-Processing Functions ---
def process_captioning(data, instruction, model):
    """
    Process image captioning:
    For each entry, generate captions for different image variations.
    """
    for entry in tqdm(data, desc="Captioning images"):
        # Skip entries without an image path or missing files
        image_path = entry.get("image")
        if not image_path:
            continue
        # Generate the caption using the image API
        entry["general_caption"] = perform_image_caption(model, image_path, instruction)
    return data


def process_caption_scoring(data, instruction, model):
    """
    Process caption scoring:
    For each entry, score existing captions for different image variations.
    """
    for entry in tqdm(data, desc="Scoring captions"):
        # Concatenate multi-turn conversation into a single string
        conversation_str = concat_conversations(entry["conversations"])

        # It is assumed that captions are already generated in the entry.
        caption = f"General caption: {entry['general_caption']}\nDetail caption: {entry.get('task_caption', '')}"
        entry["caption_score"] = perform_caption_scoring(
            model, conversation_str, caption, instruction)
    return data


def process_task_prediction(data, instruction, model):
    for entry in tqdm(data, desc="Task prediction"):
        # Concatenate multi-turn conversation into a single string
        conversation_str = concat_conversations(entry["conversations"])

        entry["predicted_task"] = perform_task_prediction(model, conversation_str, instruction)
    return data


def process_task_captioning(data, instruction, model):
    for entry in tqdm(data, desc="Task captioning"):
        # Skip entries without an image path or missing files
        image_path = entry.get("image")
        if not image_path:
            continue
        # Generate task-specific caption using the image API
        entry["task_caption"] = perform_task_caption(model, entry["sub_task"], image_path, instruction)
    return data


def process_image_quality(data, instruction, model):
    """
    Process image quality assessment:
    For each entry, assess image quality using the image API.
    """
    # Define allowed quality labels and numeric mapping
    quality_map = {
        "The quality of the image is bad.": 1,
        "The quality of the image is poor.": 2,
        "The quality of the image is fair.": 3,
        "The quality of the image is good.": 4,
        "The quality of the image is excellent.": 5,
    }
    for entry in tqdm(data, desc="Assessing image quality"):
        # Skip entries without an image path or missing files
        image_path = entry.get("image")
        if not image_path:
            continue
        # Generate image quality assessment using the image API
        assessment = perform_image_quality(model, image_path, instruction)
        if assessment not in quality_map:
            entry["image_quality_score"] = 0
            print(f"Unexpected quality assessment: {assessment}")
            continue
        entry["image_quality_score"] = quality_map[assessment]
    return data


def process_text_quality(data, instruction, model):
    """
    Process text quality assessment:
    For each entry, assess text quality using the text API.
    """
    for entry in tqdm(data, desc="Assessing text quality"):
        # Concatenate multi-turn conversation into a single string
        conversation_str = concat_conversations(entry["conversations"])

        # Extract the text to assess; adjust the key if your entries use a different field
        assessment = perform_text_quality(model, conversation_str, instruction)
        entry["text_quality_score"] = assessment
    return data


# --- Task Processor Mapping ---
TASK_PROCESSORS = {
    "captioning": process_captioning,
    "score_caption_question": process_caption_scoring,
    "predict_task": process_task_prediction,
    "task_captioning": process_task_captioning,
    "image_quality_assessment": process_image_quality,
    "text_quality_assessment": process_text_quality
}


def get_instruction_for_task(task, instructions):
    """Retrieve the instruction prompt for a given task."""
    if task in instructions:
        return instructions[task]
    else:
        return None


def main():
    args = parse_args()
    # Override the global CHAT_URL if provided via argument
    global CHAT_URL
    CHAT_URL = args.chat_url
    # Override the global IMAGE_BASE if provided via argument
    global IMAGE_BASE
    IMAGE_BASE = args.image_base

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
