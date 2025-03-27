import json

# Load the JSON data from ai_accuracy/all.json
with open("ai_accuracy/all.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract only summary and predicted_reason
filtered_data = [{"human_reason": item["human_reason"], "predicted_reason": item["predicted_reason"]} for item in data]

# Save to just_reasons.json
with open("ai_accuracy/just_reasons.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4)

print("just_reasons.json has been created successfully!")
