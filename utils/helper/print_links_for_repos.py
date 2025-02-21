import os
import json
from repos import repos
from link_extractor import extract_all_links

def extend_links_from_scraped_data(issues_file, output_file):
    """Print what links were found issues with pull request data and review comments."""
    try:
        with open(issues_file, 'r') as f:
            issues = json.load(f)
    except Exception as e:
        print(f"Error reading {issues_file}: {e}")
        return
    counter = 0
    print("Started work...")
    all_links = {}
    for issue in issues:
        is_pr = False
        if 'pull_request' in issue and 'url' in issue['pull_request']:
            is_pr = True
        links = extract_all_links(issue, is_pr)
        if links:
            all_links[issue["number"]] = links
        counter += 1
        if counter % 1000 == 0:
            print(f"Processed {counter} issues...")
        
    # Save extracted links to a JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(all_links, f, indent=2)
        print(f"Saved extracted links to {output_file}")
    except Exception as e:
        print(f"Error saving extracted links: {e}")
        
    return all_links

if __name__ == "__main__":
    for repo in repos:
        repo_name = repo.replace("/", "_")
        issues_file = os.path.join(os.getcwd(), "scraped_data", f"{repo_name}.json")
        output_file = os.path.join(os.getcwd(), "example_data", f"{repo_name}_links_only.json")
        
        all_links = extend_links_from_scraped_data(issues_file, output_file)

        print(len(all_links))
        # print(all_links)