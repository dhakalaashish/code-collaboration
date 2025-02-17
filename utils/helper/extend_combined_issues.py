import os
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_data(url, headers):
    """Fetch JSON data from a given URL with rate limit handling."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check rate limit
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 1:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - time.time(), 0)
            print(f"Rate limit reached. Waiting {sleep_time} seconds...")
            time.sleep(sleep_time)
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def enrich_issues_with_pr_data(issues_file, output_file, headers):
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
        if 'pull_request' in issue and 'url' in issue['pull_request']:
            pr_url = issue['pull_request']['url']
            pr_data = fetch_data(pr_url, headers)
            
            if pr_data:
                issue['pull_request_url_body'] = pr_data
                
                # Fetch review comments if available
                review_comments_url = pr_data.get('review_comments_url')
                if review_comments_url:
                    review_comments_data = fetch_data(review_comments_url, headers)
                    issue['pull_request_url_body']['review_comments_url_body'] = review_comments_data
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} issues...")
        
    
    try:
        with open(output_file, 'w') as f:
            json.dump(issues, f, indent=2)
        print(f"Enriched data saved to {output_file}")
    except Exception as e:
        print(f"Error saving enriched data: {e}")

if __name__ == "__main__":
    issues_file = os.path.join(os.getcwd(), "scraped_data", "combined_issues.json")
    output_file = os.path.join(os.getcwd(), "scraped_data", "combined_issues_enriched.json")
    
    github_token = os.getenv('GITHUB_AUTH_TOKEN')
    if not github_token:
        raise ValueError("GITHUB_AUTH_TOKEN not found in .env file")
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}"
    }
    
    enrich_issues_with_pr_data(issues_file, output_file, headers)