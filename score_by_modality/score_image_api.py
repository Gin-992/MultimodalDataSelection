import base64
import json
import os
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm


def encode_image_to_base64(image_path: str) -> str:
    """
    Read an image file and return a data URI string in base64 format.
    Assumes JPEG format; modify the MIME type if needed.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def chat(
        model_name: str,
        system_message: str,
        query: str,
        image: Optional[str],
        max_tokens: int,
        temperature: float,
        url: str,
        local_image_folder: str
) -> Dict[str, Any]:
    """
    Send a prompt (with optional image) to a language model via HTTP POST.

    Args:
        model_name (str): The model identifier.
        system_message (str): The system prompt.
        query (str): The combined conversation text.
        image (Optional[str]): The image filename or URL.
        max_tokens (int): Maximum tokens for generation.
        temperature (float): Temperature for generation.
        url (str): API endpoint URL.
        local_image_folder (str): Local folder path for image files.

    Returns:
        Dict[str, Any]: The model's JSON response.
    """
    if image:
        if image.startswith("http"):
            image_url = image
        else:
            # Construct full path if not absolute
            full_path = image if os.path.isabs(image) else os.path.join(local_image_folder, image)
            image_url = encode_image_to_base64(full_path)
        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": query},
            ],
        }
    else:
        user_message = {"role": "user", "content": query}

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            user_message
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error during API request: {e}") from e


def load_json(file_path: str) -> Any:
    """Load JSON data from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """Save data as JSON to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def combine_conversations(conversations: List[Dict[str, str]]) -> str:
    """
    Combine conversation turns into a single prompt string.
    Each "human" message is prefixed with "### user:" and each "gpt" message with "### assistant:".
    The prompt ends with "### Rating:".

    Args:
        conversations (List[Dict[str, str]]): List of conversation turns.

    Returns:
        str: The combined conversation string.
    """
    lines = []
    for conv in conversations:
        sender = conv.get("from", "").strip().lower()
        value = conv.get("value", "").strip()
        if sender == "human":
            lines.append(f"### user: {value}")
        elif sender == "gpt":
            lines.append(f"### assistant: {value}")
    lines.append("### Rating:")
    return "\n".join(lines)


def process_entries(
        local_image_folder: str,
        data_entries: List[Dict[str, Any]],
        model_name: str,
        system_message: str,
        url: str,
        max_tokens: int,
        temperature: float,
) -> Dict[str, Dict[str, str]]:
    """
    Process each data entry by combining its conversations and sending the prompt (with image) to the model.

    Args:
        local_image_folder (str): Folder where local images are stored.
        data_entries (List[Dict[str, Any]]): List of data entry dictionaries.
        model_name (str): The model identifier.
        system_message (str): The system prompt.
        url (str): API endpoint URL.
        max_tokens (int): Maximum tokens for generation.
        temperature (float): Temperature for generation.

    Returns:
        Dict[str, Dict[str, str]]: Mapping from entry IDs to their image and rating output.
    """
    results = {}
    for entry in tqdm(data_entries):
        entry_id = entry.get("id")
        image = entry.get("image")
        conversations = entry.get("conversations", [])
        combined_prompt = combine_conversations(conversations)

        ret_msg = chat(
            model_name,
            system_message,
            combined_prompt,
            image,
            max_tokens,
            temperature,
            url,
            local_image_folder
        )
        model_msg = ret_msg["choices"][0]["message"]["content"]

        print(f"Entry ID: {entry_id}")
        print(f"Image: {image}")
        print("Model Rating Output:")
        print(model_msg)
        print("-" * 80)

        results[entry_id] = {"image": image, "rating": model_msg}

    return results


def main() -> None:
    # Fixed system prompt for evaluation.
    iqa_system_message = (
        "As a data quality estimator, your task is to assess the quality of a data sample based on the following criteria: "
        "Sharpness, noise, color accuracy and exposure. Please rate the sample on a scale from 1 to 10 for each criterion, and return "
        "an overall rating on a scale from 1 to 10 (with higher scores indicating higher quality). "
        "Return your ratings using the following JSON format:\n"
        '{\n'
        '    "Sharpness": <number, 1-10>,\n'
        '    "Noise": <number, 1-10>,\n'
        '    "Color accuracy": <number, 1-10>,\n'
        '    "Exposure": <number, 1-10>,\n'
        '    "Overall rating": <number, 1-10>\n'
        '}'
    )
    mm_system_message = (
        "As a data quality estimator, your task is to assess the quality of a data sample based on the following criteria: "
        "Grounding accuracy, context consistency and dialogue coherence. Please rate the sample on a scale from 1 to 10 for each criterion, and return "
        "an overall rating on a scale from 1 to 10 (with higher scores indicating higher quality). "
        "Return your ratings using the following JSON format:\n"
        '{\n'
        '    "Grounding accuracy": <number, 1-10>,\n'  # Measures whether textual references correctly “ground” in the actual content of the image (e.g., no hallucination of objects that are not present).
        '    "Context consistency": <number, 1-10>,\n'  # Evaluates whether the dialogue maintains a logical and consistent reference to the image across turns.
        '    "Dialogue coherence": <number, 1-10>,\n'  # Score the natural flow, relevance, and context continuity of multi-turn dialogues.
        '    "Overall rating": <number, 1-10>\n'
        '}'
    )

    # Configuration parameters.
    url = "http://localhost:1234/v1/chat/completions"  # Update to your API endpoint.
    max_tokens = 1024
    temperature = 0.1
    model_name = "qwen2-vl-7b-instruct"  # Update if using a different model.
    json_file_path = "/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_llava_complex_reasoning_annotations.json"  # Path to your input JSON file.
    # Folder where local images are stored (adjust as needed)
    local_image_folder = "/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_complex_reasoning_images"

    data_entries = load_json(json_file_path)
    results = process_entries(local_image_folder, data_entries, model_name, mm_system_message, url, max_tokens, temperature)

    output_file = "../rating_scores/llava-conversations/llava_conversation_image_mm_ratings_output.json"
    save_json(results, output_file)
    print(f"Ratings saved to {output_file}")


if __name__ == "__main__":
    main()
