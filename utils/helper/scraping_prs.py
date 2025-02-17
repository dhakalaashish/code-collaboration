import requests
from time import sleep
import time
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
prs_per_page = 100
start_page = 111
current_batch = 12

print('Started Scraping Pull Requests')

def save_to_json(data, batch_number):
    """Save PR data to a JSON file in scraped_data/prs/"""
    save_dir = os.path.join(os.getcwd(), "scraped_data", "prs")  # Move one directory up, then into scraped_data/prs
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
    filename = os.path.join(save_dir, f'prs_batch{batch_number}.json')
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def fetch_additional_data(pr, headers):
    """Fetch review comments from review_comments_url"""
    try:
        # Fetch review comments
        review_comments_response = requests.get(pr['review_comments_url'], headers=headers, timeout=10)
        review_comments_response.raise_for_status()
        pr['review_comments_body'] = review_comments_response.json()
        # Check rate limit
        remaining = int(review_comments_response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 1:
            reset_time = int(review_comments_response.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - time.time(), 0)
            print(f"Rate limit reached. Waiting {sleep_time} seconds...")
            sleep(sleep_time)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching review comments for PR #{pr['number']}: {e}")
        pr['review_comments_body'] = []

    return pr

# GitHub API URL for Pull Requests
repo_owner = "jax-ml"
repo_name = "jax"
url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"

params = {
    "state": "closed",  # Fetch closed PRs
    "per_page": prs_per_page,
    "page": start_page
}

# Get GitHub Token
github_token = os.getenv('GITHUB_AUTH_TOKEN')
if not github_token:
    raise ValueError("GITHUB_AUTH_TOKEN not found in .env file")

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {github_token}"
}

all_prs = []
pagesLeft = True

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
        if not data:  # No more PRs
            break

        # Fetch additional data for each PR
        for pr in data:
            fetch_additional_data(pr, headers)

        # Check if we got less than 100 PRs
        if len(data) < prs_per_page:
            print("No more pages left in PRs")
            pagesLeft = False

        # Add the PRs to all_prs
        all_prs.extend(data)
        params["page"] += 1
        print(f"Page {params['page']-1} processed")

        # Save all_prs to json every 10 pages
        if (params["page"]-1) % 10 == 0:
            print(f"Saving batch {current_batch} (pages {(current_batch-1)*10 + 1}-{(current_batch)*10})")
            save_to_json(all_prs, current_batch)
            all_prs = []  # Clear all_prs list after saving
            current_batch += 1

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        break

# Save remaining PRs at the end if any
if all_prs:
    print(f"Saving final batch {current_batch}...")
    save_to_json(all_prs, current_batch)

print("Scraping completed")
print(f"Total batches created: {current_batch}")


