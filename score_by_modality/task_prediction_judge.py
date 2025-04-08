import json
import argparse

# Mapping dictionaries
l2_task_mapping = {
    "Image Style": "image_style",
    "Image Scene": "image_scene",
    "Image Emotion": "image_emotion",
    "Image Quality": "image_quality",
    "Image Topic": "image_topic",
    "Object Localization": "object_localization",
    "Attribute Recognition": "attribute_recognition",
    "Celebrity Recognition": "celebrity_recognition",
    "OCR (Optical Character Recognition)": "ocr",
    "Spatial Relationship": "spatial_relationship",
    "Attribute Comparison": "attribute_comparison",
    "Action Recognition": "action_recognition",
    "Physical Property Reasoning": "physical_property_reasoning",
    "Function Reasoning": "function_reasoning",
    "Identity Reasoning": "identity_reasoning",
    "Social Relation": "social_relation",
    "Physical Relation": "physical_relation",
    "Nature Relation": "nature_relation",
    "Structuralized Image-Text Understanding": "structuralized_imagetext_understanding",
    "Future Prediction": "future_prediction"
}

l1_task_mapping = {
    "Coarse Perception": "coarse_perception",
    "Fine-grained Perception (single-instance)": "finegrained_perception (instance-level)",
    "Fine-grained Perception (cross-instance)": "finegrained_perception (cross-instance)",
    "Attribute Reasoning": "attribute_reasoning",
    "Relation Reasoning": "relation_reasoning",
    "Logic Reasoning": "logic_reasoning",
}


def evaluate_predictions(json_file):
    """
    Reads the specified JSON file, then for each data entry:
      - Extracts the ground truth category (metadata["category"]) and l2_category (metadata["l2_category"]).
      - Reads the predicted task (field "task") and predicted sub-task (field "sub-task").
      - Normalizes predictions using mapping dictionaries.
      - Compares:
           * The normalized predicted task (using l1_task_mapping) vs. ground truth l2_category.
           * The normalized predicted sub-task (using l2_task_mapping) vs. ground truth category.

      Additionally, calculates and prints:
         - Overall accuracy for both comparisons.
         - Per-class accuracy for each ground truth label.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    task_correct = 0
    sub_task_correct = 0

    # Dictionaries to track counts for per-class accuracy.
    # For task predictions: key = ground truth l2_category
    task_stats = {}
    # For sub-task predictions: key = ground truth category
    sub_stats = {}

    for item in data:
        # Ground truth values (using .strip() to clean whitespace)
        gt_category = item.get("metadata", {}).get("category", "").strip()
        gt_l2_category = item.get("metadata", {}).get("l2_category", "").strip()

        # Predicted values
        pred_task_raw = item.get("task", "").strip()  # predicted task
        pred_sub_task_raw = item.get("sub-task", "").strip()  # predicted sub-task

        # Normalize predictions using the mapping dictionaries.
        normalized_pred_task = l1_task_mapping.get(pred_task_raw, pred_task_raw)
        normalized_pred_sub = l2_task_mapping.get(pred_sub_task_raw, pred_sub_task_raw)

        # Initialize counters for the ground truth labels if necessary.
        if gt_l2_category not in task_stats:
            task_stats[gt_l2_category] = {"total": 0, "correct": 0}
        if gt_category not in sub_stats:
            sub_stats[gt_category] = {"total": 0, "correct": 0}

        # Update counters for task predictions (predicted task vs ground truth l2_category).
        task_stats[gt_l2_category]["total"] += 1
        if normalized_pred_task.lower() == gt_l2_category.lower():
            task_stats[gt_l2_category]["correct"] += 1
            task_correct += 1

        # Update counters for sub-task predictions (predicted sub-task vs ground truth category).
        sub_stats[gt_category]["total"] += 1
        if normalized_pred_sub.lower() == gt_category.lower():
            sub_stats[gt_category]["correct"] += 1
            sub_task_correct += 1

    # Calculate overall accuracies.
    task_accuracy = task_correct / total if total > 0 else 0
    sub_task_accuracy = sub_task_correct / total if total > 0 else 0
    combined_accuracy = (task_correct + sub_task_correct) / (2 * total) if total > 0 else 0

    print("Overall Accuracies:")
    print("-------------------------------------------------")
    print("Task Accuracy (predicted task vs. l2_category): {:.2f}%".format(task_accuracy * 100))
    print("Sub-task Accuracy (predicted sub-task vs. category): {:.2f}%".format(sub_task_accuracy * 100))
    print("Combined Accuracy: {:.2f}%".format(combined_accuracy * 100))
    print("\nPer-Class Accuracies:")

    print("\n--- Task Prediction (Ground Truth l2_category) ---")
    for cls, stats in task_stats.items():
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print("  {}: {:.2f}% ({} out of {})".format(cls, accuracy, stats["correct"], stats["total"]))

    print("\n--- Sub-task Prediction (Ground Truth category) ---")
    for cls, stats in sub_stats.items():
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print("  {}: {:.2f}% ({} out of {})".format(cls, accuracy, stats["correct"], stats["total"]))


if __name__ == "__main__":
    # Using argparse to allow input of the JSON file path from the command line.
    parser = argparse.ArgumentParser(
        description="Evaluate the accuracy of task prediction including per-class analysis")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing predictions")
    args = parser.parse_args()

    evaluate_predictions(args.json_file)