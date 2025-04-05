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

    return model_resp


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

        ret_msg = chat(
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

"""You are provided with a text question. Your task is to predict which specific task and sub-task should be performed based on the question alone. The tasks are divided into several categories, and each category has specific definitions to guide you. Use these definitions to identify the appropriate task and sub-task.

Task Categories and Definitions:
1. Coarse Perception:
    (1) Image Style: Determine the type of image (e.g., photograph, painting, CT scan, etc.)
    (2) Image Scene: Identify the environment (e.g., indoors, outdoors, forest, city, etc.)
    (3) Image Emotion: Recognize the subjective emotion conveyed (e.g., cheerful, sad, oppressive)
    (4) Image Quality: Assess the image quality (e.g., blurry, bright, dark, high contrast)
    (5) Image Topic: Identify the subject of the image (e.g., portrait, scenery, close-up of an object)
2. Fine-grained Perception (single-instance):
    (1) Object Localization: Determine the position and orientation of a single object in the image
    (2) Attribute Recognition: Recognize attributes such as shape, texture, or appearance
    (3) Celebrity Recognition: Recognize well-known personalities, landmarks, or famous objects
    (4) OCR (Optical Character Recognition): Extract and recognize text, formulas, or sheets present in the image
3. Fine-grained Perception (cross-instance):
    (1) Spatial Relationship: Determine the relative positions between multiple objects
    (2) Attribute Comparison: Compare attributes of different objects (e.g., size, shape, color)
    (3) Action Recognition: Recognize human actions, such as pose, movement, or interactions between humans and objects
4. Attribute Reasoning:
    (1) Physical Property Reasoning: Predict physical properties of an object (e.g., fluidity of water, volatility of sulfuric acid)
    (2) Function Reasoning: Predict the function or use of an object (e.g., broom for sweeping, pen for writing)
    (3) Identity Reasoning: Predict the identity or role of a person or object based on appearance (e.g., occupation based on clothing)
5. Relation Reasoning:
    (1) Social Relation: Identify relationships between humans (e.g., father and son, husband and wife, friend)
    (2) Physical Relation: Describe spatial and physical relationships between objects (e.g., above, below, in contact)
    (3) Nature Relation: Identify abstract relationships in nature (e.g., predation, symbiosis, coexistence)
6. Logic Reasoning:
    (1) Structuralized Image-Text Understanding: Interpret structured data in images with text, such as charts or formulas
    (2) Future Prediction: Predict future events or outcomes based on current information (e.g., weather change, emotional shift)

Steps to Follow:
1. Review the Question: Read the question carefully and understand what information is being asked for.
2. Identify the Task Category: Based on the definitions above, categorize the task into one of the six categories.
3. Select the Sub-task: Choose the most appropriate sub-task from the selected category based on the specific details of the question.
4. Output the Task and Sub-task: Predict the task category and the corresponding sub-task.

Example Input:
    Question: "What is the relationship between the dog and the cat in the image?"
Output:
    {"task": "Fine-grained Perception (cross-instance)", "sub-task": "Spatial Relationship"}
"""


"""You are provided with an image. Your task is to generate a caption that predicts a future event or outcome based on the current visual cues in the image. Focus on identifying trends, patterns, or signs that indicate a forthcoming change, such as a weather shift, an evolving atmosphere, or an anticipated event. Your caption should logically project what might happen next based on the present state depicted in the image.

Guidelines:

1. Examine the Image:
    (1) Carefully observe the current state of the image.
    (2) Identify any visual cues or trends that could suggest an imminent change (e.g., darkening clouds, shifting colors, facial expressions).
2. Identify Clues for Future Change:
    (1) Look for indicators such as environmental transitions (e.g., the sky darkening might hint at an approaching storm) or emotional cues (e.g., a person's serious expression might suggest a forthcoming emotional shift).
3. Predict the Future Outcome:
    (1) Based on the observed cues, infer a likely future event or outcome.
    (2) Ensure that the prediction logically follows from the current visual information.
4. Craft a Concise Caption:
    (1) Write a single sentence caption that clearly communicates the predicted outcome.
    (2) Focus solely on the future event or change without including unrelated details.

Your final caption should succinctly capture the anticipated future event or outcome based on the visual information provided in the image.
"""