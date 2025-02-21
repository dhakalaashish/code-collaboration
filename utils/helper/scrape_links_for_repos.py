import os
import json
from repos import repos
from link_extractor import extract_all_links

def extend_links_from_scraped_data(issues_file, output_file):
    """Enrich issues with pull request data and review comments."""
    try:
        with open(issues_file, 'r') as f:
            issues = json.load(f)
    except Exception as e:
        print(f"Error reading {issues_file}: {e}")
        return
    counter = 0
    print("Started work...")
    for issue in issues:
        is_pr = False
        if 'pull_request' in issue and 'url' in issue['pull_request']:
            is_pr = True
        links = extract_all_links(issue, is_pr)
        if links:
            issue['links_to'] = links
        counter += 1
        if counter % 1000 == 0:
            print(f"Processed {counter} issues...")
        
    try:
        with open(output_file, 'w') as f:
            json.dump(issues, f, indent=2)
        print(f"Enriched data saved to {output_file}")
    except Exception as e:
        print(f"Error saving enriched data: {e}")

if __name__ == "__main__":
    for repo in repos:
        repo_name = repo.replace("/", "_")
        issues_file = os.path.join(os.getcwd(), "scraped_data", f"{repo_name}.json")
        output_file = os.path.join(os.getcwd(), "scraped_data", f"{repo_name}_with_links.json")
        
        extend_links_from_scraped_data(issues_file, output_file)

        print("Extending links completed!")