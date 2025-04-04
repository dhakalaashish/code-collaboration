import os
import json
from repos import repos  # Import repo list from the original script

def merge_json_files(repo):
    """Merge all JSON pages for a given repository."""
    repo_path = repo.replace("/", "_")
    input_dir = os.path.join("scraped_issues", repo_path)
    output_file = os.path.join(input_dir, f"{repo_path}_merged.json")
    
    if not os.path.exists(input_dir):
        print(f"No data found for {repo}. Skipping...")
        return
    
    all_issues = []
    
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith(".json") and "merged" not in file_name:
            file_path = os.path.join(input_dir, file_name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_issues.extend(data)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    try:
        with open(output_file, "w") as f:
            json.dump(all_issues, f, indent=2)
        print(f"Merged data saved to {output_file}")
    except Exception as e:
        print(f"Error saving merged data for {repo}: {e}")

def main():
    """Main function to merge JSON files for all repositories."""
    
    for repo in repos:
        print(f"Merging pages for {repo}...")
        merge_json_files(repo)
    
    print("Merging completed.")

if __name__ == "__main__":
    main()
