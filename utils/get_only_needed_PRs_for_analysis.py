import os
import json

# Define input and output directories
INPUT_DIR = "ai_generated_15_reasons/repos"
OUTPUT_DIR = "ai_generated_15_reasons/analysis/repos"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def filter_json_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):  # Process only JSON files
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Keep only the first 14 entries
            filtered_data = data[:14]

            # Retain "url" and "predicted_reason", and add empty human_reason fields
            processed_data = [
                {
                    "url": entry["url"],
                    "predicted_reason": entry["predicted_reason"],
                    "human_reason_1": [],
                    "human_reason_2": []
                }
                for entry in filtered_data
            ]

            # Save the modified JSON to the output directory
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=4)

            print(f"Processed {filename}: Kept {len(processed_data)} entries. Saved to {output_file_path}")

# Run the filtering process
if __name__ == "__main__":
    filter_json_files(INPUT_DIR, OUTPUT_DIR)