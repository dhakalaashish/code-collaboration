import os
import json
import random

# Load clustered issues
clustered_issues_file = os.path.join(os.path.dirname(__file__), "clustered_issues.json")
selected_issues_file = "selected_issues.json"

if not os.path.exists(clustered_issues_file):
    print(f"Error: File '{clustered_issues_file}' not found.")
else:
    with open(clustered_issues_file, "r", encoding="utf-8") as f:
        clustered_issues = json.load(f)

    # Dictionary to store selected PRs
    selected_issues = {}

    # Select one random PR from each cluster
    for cluster, issues in clustered_issues.items():
        if issues:  # Ensure the cluster has issues
            selected_issues[cluster] = random.choice(issues)  # Select one PR

    # Save selected issues to a JSON file
    with open(selected_issues_file, "w", encoding="utf-8") as f:
        json.dump(selected_issues, f, indent=4)

    print(f"Selected issues saved to {selected_issues_file}")
