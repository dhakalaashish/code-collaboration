import os
import json

# Directories for input and output
INPUT_DIR = "ai_generated_15_reasons/analysis/repos"
OUTPUT_FILE = "ai_generated_15_reasons/analysis/jaccard_results.json"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0  # Both empty, considered identical
    if not set1 or not set2:
        return 0.0  # One is empty, no similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def analyze_similarity(input_dir):
    """Compute similarity scores across multiple JSON files."""
    all_human_agreement = []
    all_human_vs_ai = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(input_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            human_agreement = []
            human_vs_ai = []

            for entry in data:
                set_h1 = set(entry.get("human_reason_1", []))
                set_h2 = set(entry.get("human_reason_2", []))
                set_ai = set(entry.get("predicted_reason", []))

                # Combine both human annotations
                combined_human_set = set_h1 | set_h2  # Union of both human annotations

                # Calculate Jaccard Similarities
                human_agreement.append(jaccard_similarity(set_h1, set_h2))
                human_vs_ai.append(jaccard_similarity(combined_human_set, set_ai))

            # Aggregate across all files
            all_human_agreement.extend(human_agreement)
            all_human_vs_ai.extend(human_vs_ai)

    # Compute overall average scores
    avg_human_agreement = sum(all_human_agreement) / len(all_human_agreement) if all_human_agreement else 0
    avg_human_vs_ai = sum(all_human_vs_ai) / len(all_human_vs_ai) if all_human_vs_ai else 0

    results = {
        "average_human_agreement": avg_human_agreement,
        "average_human_vs_ai": avg_human_vs_ai
    }

    # Save results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Jaccard similarity results saved to {OUTPUT_FILE}!")

if __name__ == "__main__":
    analyze_similarity(INPUT_DIR)