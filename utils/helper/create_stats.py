import os
import json

# Get the absolute path to the HCI-RA directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

# Path to the scraped_issues directory
scraped_issues_dir = os.path.join(base_dir, "scraped_issues")  

# Path to output file dir
output_dir = os.path.join(base_dir, "helper_data")

# Output file path
output_file_issues_without_pr = os.path.join(output_dir, "issues_id_without_PR.json")
output_file_issues_with_pr = os.path.join(output_dir, "issues_id_with_PR.json")

# List to store issues without a "pull_request" key
issues_without_pr = []
issues_with_pr = []

# Loop through all JSON files in the directory
for filename in os.listdir(scraped_issues_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(scraped_issues_dir, filename)
        try:
            with open(filepath, "r") as f:
                issues = json.load(f)
                for issue in issues:
                    if "pull_request" not in issue:
                        issues_without_pr.append(issue["id"])
                    else:
                        issues_with_pr.append(issue["id"])
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Save the filtered issues to a new JSON file
if issues_without_pr:
    with open(output_file_issues_without_pr, "w") as f:
        json.dump(issues_without_pr, f, indent=2)
    print(f"Saved {len(issues_without_pr)} issues without pull requests to {output_file_issues_without_pr}")
else:
    print("No issues without PR found.")

if issues_with_pr:
    with open(output_file_issues_with_pr, "w") as f:
        json.dump(issues_with_pr, f, indent=2)
    print(f"Saved {len(issues_with_pr)} issues with pull requests to {output_file_issues_with_pr}")
else:
    print("No issues with PR found.")