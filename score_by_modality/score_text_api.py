import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


def chat(
    model_name: str,
    system_message: str,
    query: str,
    max_tokens: int,
    temperature: float,
    url: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Send a prompt to a language model via HTTP POST.

    Args:
        model_name (str): The model identifier.
        system_message (str): The system prompt message.
        query (str): The user prompt (combined conversation string).
        max_tokens (int): Maximum number of tokens for the response.
        temperature (float): Temperature parameter.
        url (str): API endpoint URL.

    Returns:
        Tuple[Optional[str], Dict[str, Any]]: Returns None for image and the model's response.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        model_resp = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error during API request: {e}") from e

    return None, model_resp


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
    data_entries: List[Dict[str, Any]],
    model_name: str,
    system_message: str,
    url: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Dict[str, str]]:
    """
    Process each data entry by combining its conversations and sending the prompt to the model.

    Args:
        data_entries (List[Dict[str, Any]]): List of data entry dictionaries.
        model_name (str): The model identifier.
        system_message (str): The system prompt.
        url (str): API endpoint URL.
        max_tokens (int): Maximum tokens for generation.
        temperature (float): Temperature for generation.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping entry IDs to their image and rating output.
    """
    results = {}
    for entry in tqdm(data_entries):
        entry_id = entry.get("id")
        image = entry.get("image")
        conversations = entry.get("conversations", [])
        combined_prompt = combine_conversations(conversations)

        _, ret_msg = chat(
            model_name,
            system_message,
            combined_prompt,
            max_tokens,
            temperature,
            url,
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
    # Define a fixed system prompt for evaluation.
    system_message = (
        "As a data quality estimator, your task is to assess the quality of a data sample based on the following criteria: "
        "Rarity, Complexity, and Informativeness. Please rate the sample on a scale from 1 to 10 for each criterion, and return "
        "an overall rating on a scale from 1 to 10 (with higher scores indicating higher quality). "
        "Return your ratings using the following JSON format:\n"
        '{\n'
        '    "Rarity": <number, 1-10>,\n'
        '    "Complexity": <number, 1-10>,\n'
        '    "Informativeness": <number, 1-10>,\n'
        '    "Overall rating": <number, 1-10>\n'
        '}'
    )

    # Configuration parameters
    url = "http://localhost:1234/v1/chat/completions"
    max_tokens = 1024
    temperature = 0.1
    model_name = "qwen2.5-14b-instruct"
    json_file_path = "/Volumes/Ming-Data/Dataset/Sample-MM-Data/sampled_llava_conversation_annotations.json"  # Adjust the file path as needed

    # Load data entries
    data_entries = load_json(json_file_path)
    # Process all entries and get the ratings
    results = process_entries(data_entries, model_name, system_message, url, max_tokens, temperature)

    # Save results to an output JSON file
    output_file = "llava_conversation_ratings_output.json"
    save_json(results, output_file)
    print(f"Ratings saved to {output_file}")


if __name__ == "__main__":
    main()
