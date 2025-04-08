import json
import regex as re
import statistics

with open("/output/task_prediction.json", "r") as f:
    data = json.load(f)
# pattern = r'\{.*?\}'
pattern = r'(?s)(\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\])'
scores = []
for item in data:
    predicted_task = item['predicted_task']
    predicted_task = predicted_task.split("</think>")[-1]
    matched = re.search(pattern, predicted_task)
    if matched:
        json_str3 = matched.group()
        json_str3.replace("\'", "\"")
        rating = json.loads(json_str3)
        item['task'] = rating['task']
        item['sub-task'] = rating['sub-task']
    scores.append(item)
    # untouched_score_raw = item['untouched_score']
    # matched = re.search(pattern, untouched_score_raw)
    # if matched:
    #     json_str1 = matched.group()
    #     rating = json.loads(json_str1)
    #     item['untouched_rating'] = rating['rating']
    #
    # gaussian_score_raw = item['gaussian_score']
    # matched = re.search(pattern, untouched_score_raw)
    # if matched:
    #     json_str2 = matched.group()
    #     rating = json.loads(json_str2)
    #     item['gaussian_rating'] = rating['rating']
    #
    # salt_pepper_score_raw = item['salt_pepper_score']
    # matched = re.search(pattern, salt_pepper_score_raw)
    # if matched:
    #     json_str3 = matched.group()
    #     rating = json.loads(json_str3)
    #     item['salt_pepper_rating'] = rating['rating']
    # scores.append(item)
with open("/output/task_prediction_clean.json", "w") as f:
    json.dump(scores, f, indent=4, default=str)

# Filter out entries with any missing rating value and count missing fields
valid_entries = []
missing_count = 0
rating_fields = ["untouched_rating", "gaussian_rating", "salt_pepper_rating"]

for entry in scores:
    missing_in_entry = sum(1 for field in rating_fields if entry.get(field) is None)
    if missing_in_entry > 0:
        missing_count += missing_in_entry
        continue  # discard this entry entirely
    valid_entries.append(entry)

print(f"Total missing rating fields found (and discarded): {missing_count}\n")

# If there are no valid entries, exit
if not valid_entries:
    print("No valid entries available for analysis.")
    exit()

# Extract ratings from valid entries
untouched_ratings = [entry["untouched_rating"] for entry in valid_entries]
gaussian_ratings  = [entry["gaussian_rating"] for entry in valid_entries]
salt_pepper_ratings = [entry["salt_pepper_rating"] for entry in valid_entries]

# Helper function to calculate median and standard deviation
def calc_stats(ratings):
    med = statistics.median(ratings)
    # Using sample standard deviation; if you prefer population stdev use statistics.pstdev
    std = statistics.stdev(ratings) if len(ratings) > 1 else 0
    return med, std

untouched_med, untouched_std = calc_stats(untouched_ratings)
gaussian_med, gaussian_std   = calc_stats(gaussian_ratings)
salt_pepper_med, salt_pepper_std = calc_stats(salt_pepper_ratings)

# Calculate shifts for each valid entry:
# shift_ug = untouched_rating - gaussian_rating
# shift_us = untouched_rating - salt_pepper_rating
shifts_ug = [entry["untouched_rating"] - entry["gaussian_rating"] for entry in valid_entries]
shifts_us = [entry["untouched_rating"] - entry["salt_pepper_rating"] for entry in valid_entries]

# Calculate average shifts
avg_shift_ug = sum(shifts_ug) / len(shifts_ug) if shifts_ug else 0
avg_shift_us = sum(shifts_us) / len(shifts_us) if shifts_us else 0

# Display the results
print("Rating Statistics:")
print("------------------")
print(f"Untouched Rating: median = {untouched_med}, standard deviation = {untouched_std}")
print(f"Gaussian Rating:  median = {gaussian_med}, standard deviation = {gaussian_std}")
print(f"Salt & Pepper Rating: median = {salt_pepper_med}, standard deviation = {salt_pepper_std}\n")

print("Average Shifts:")
print("---------------")
print(f"Average shift (untouched - gaussian) = {avg_shift_ug}")
print(f"Average shift (untouched - salt_pepper) = {avg_shift_us}")

# For each valid entry, also print the individual shifts if needed:
print("\nIndividual Entry Shifts:")
for i, entry in enumerate(valid_entries, start=1):
    shift_ug = entry["untouched_rating"] - entry["gaussian_rating"]
    shift_us = entry["untouched_rating"] - entry["salt_pepper_rating"]
    print(f"Entry {i}: shift (untouched - gaussian) = {shift_ug}, shift (untouched - salt_pepper) = {shift_us}")