import requests
from time import sleep
import time
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def save_to_json(data, batch_number):
    """Save data to a JSON file for specific batch"""
    os.makedirs('scraped_issues', exist_ok=True)
    filename = f'scraped_issues/issues_batch{batch_number}.json'
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def fetch_additional_data(issue, headers):
    """Fetch patch and comments data for an issue"""
    if 'pull_request' in issue:
        try:
            patch_response = requests.get(issue['pull_request']['patch_url'], headers=headers, timeout=10)
            patch_response.raise_for_status()
            issue['pull_request']['patch_url_body'] = patch_response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching patch for #{issue['number']}: {e}")
            issue['pull_request']['patch_url_body'] = ""

    try:
        comments_response = requests.get(issue['comments_url'], headers=headers, timeout=10)
        comments_response.raise_for_status()
        issue['comments_url_body'] = comments_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching comments for #{issue['number']}: {e}")
        issue['comments_url_body'] = []

    return issue

url = "https://api.github.com/repos/jax-ml/jax/issues"

params = {
    "state": "closed",
    "per_page": 100,
    "page": 1
}

github_token = os.getenv('GITHUB_AUTH_TOKEN')
if not github_token:
    raise ValueError("GITHUB_AUTH_TOKEN not found in .env file")

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {github_token}"
}

# Collect all issues
all_issues = []
pagesLeft = True
current_batch = 1

while pagesLeft:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Check rate limit
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 1:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - time.time(), 0)
            print(f"Rate limit reached. Waiting {sleep_time} seconds...")
            sleep(sleep_time)
            continue
        
        data = response.json()
        if not data:  # No more issues
            break
            
        # Fetch additional data for each issue
        for issue in data:
            fetch_additional_data(issue, headers)
            
        # Check if we got less than 100 issues
        if len(data) < 100:
        # if params["page"] > 22:  #Uncomment for testing
            pagesLeft = False
            
        # Add the issues to all_issues
        all_issues.extend(data)
        params["page"] += 1
        print(f"Page {params['page']-1} processed")
        
        # Save all_issues to json every 10 pages
        if (params["page"]-1) % 10 == 0:
            print(f"Saving batch {current_batch} (pages {(current_batch-1)*10 + 1}-{(current_batch)*10})...")
            save_to_json(all_issues, current_batch)
            all_issues = []  # Clear all_issues list after saving
            current_batch += 1
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        break

# Save remaining issues at the end if any
if all_issues:
    print(f"Saving final batch {current_batch}...")
    save_to_json(all_issues, current_batch)

print("Scraping completed")
print(f"Total batches created: {current_batch}")
