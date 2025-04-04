import os
import json
from collections import defaultdict

# Directory containing AI-generated JSON files
INPUT_DIR = "ai_generated_15_reasons/analysis/repos"
OUTPUT_FILE = "ai_generated_15_reasons/analysis/reason_counts.json"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Dictionary to count occurrences of each reason (1 to 15)
reason_counts = defaultdict(int)

# Process each JSON file in the directory
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(INPUT_DIR, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # Extract and count reasons from each PR entry
            for item in data:
                predicted_reasons = item.get("predicted_reason", [])
                for reason in predicted_reasons:
                    reason_counts[reason] += 1

# Save reason count statistics
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dict(reason_counts), f, indent=4)

print(f"Reason counts saved to {OUTPUT_FILE}!")
