import os
import json
import re

def extract_batch_number(filename):
    """Extract batch number from filename using regex."""
    match = re.search(r'batch(\d+)\.json$', filename)
    return int(match.group(1)) if match else float('inf')

def combine_json_files(input_dir, output_file):
    """Combine all JSON files in a directory into a single JSON file."""
    combined_data = []
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    json_files.sort(key=extract_batch_number)  # Sort files based on batch number
    
    for filename in json_files:
        file_path = os.path.join(input_dir, filename)
        print(f"Adding file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"Combined data saved to {output_file}")
    except Exception as e:
        print(f"Error saving combined data: {e}")

if __name__ == "__main__":
    pr_input_dir = os.path.join(os.getcwd(), "scraped_data", "prs")
    pr_output_file = os.path.join(os.getcwd(), "scraped_data", "combined_prs.json")
    
    issues_input_dir = os.path.join(os.getcwd(), "scraped_data", "issues")
    issues_output_file = os.path.join(os.getcwd(), "scraped_data", "combined_issues.json")
    
    print("Combining PR JSON files...")
    combine_json_files(pr_input_dir, pr_output_file)
    print("Completed Combining PR Batches")
    
    print("Combining Issues JSON files...")
    combine_json_files(issues_input_dir, issues_output_file)
    print("Completed Combining Issues file")
