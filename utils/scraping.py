import os
import json
import time
import requests
from dotenv import load_dotenv
from link_extractor import extract_all_links
from repos import repos

# Load environment variables
load_dotenv()

CHECKPOINT_FILE = "checkpoint.json"

def load_checkpoint():
    """Load checkpoint file to resume scraping from last saved page."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(repo, page):
    """Save the last fetched page for a repository."""
    checkpoint = load_checkpoint()
    checkpoint[repo] = page
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

def fetch_data(url, headers):
    """Fetch JSON data from a given URL with rate limit handling."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Handle rate limits
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

def fetch_issues(repo, headers):
    """Fetch closed issues for a given repo and save after each page."""
    url = f"https://api.github.com/repos/{repo}/issues"
    checkpoint = load_checkpoint()
    start_page = checkpoint.get(repo, 1)  # Resume from last saved page
    
    params = {"state": "closed", "per_page": 100, "page": start_page}

    while True:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            # Handle rate limits
            remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            if remaining < 20: # always keep a buffer of 20 requests
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                sleep_time = max(reset_time - time.time(), 0)
                print(f"Rate limit reached. Waiting {sleep_time} seconds...")
                time.sleep(sleep_time)

            data = response.json()
            if not data:
                break
            
            for issue in data:
                is_pr = False
                if 'pull_request' in issue and 'url' in issue['pull_request']:
                    is_pr = True
                    pr_url = issue['pull_request']['url']
                    pr_data = fetch_data(pr_url, headers)
                    if pr_data:
                        issue['pull_request_url_body'] = pr_data
                        review_comments_url = pr_data.get('review_comments_url')
                        if review_comments_url:
                            issue['pull_request_url_body']['review_comments_url_body'] = fetch_data(review_comments_url, headers)
                        
                        # also get commit message
                        commmits_url = pr_data.get('commits_url')
                        if commmits_url:
                            commits_data = fetch_data(commmits_url, headers)
                            if commits_data:
                                issue['pull_request_url_body']['commit_message'] = commits_data[0]['commit']['message']


                # fetch comments from the comments url
                comments_url = issue['comments_url']
                if comments_url:
                    comments_data = fetch_data(comments_url, headers)
                    if comments_data:
                        issue['comments_url_body'] = comments_data
                
                # scrape the body, comments, review_comments, commit messages to find links
                links = extract_all_links(issue, is_pr)
                if links:
                    issue['links_to'] = links
            
            print(f"{repo} Page {params['page']} Done")

            # Save page immediately after processing
            save_page_issues(repo, data, params["page"]) 
            save_checkpoint(repo, params["page"] + 1)

            params["page"] += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching issues for {repo}: {e}")
            break
    return

def save_page_issues(repo, issue_list, page):
    """Save the current page's issues."""
    repo_path = repo.replace("/", "_")
    output_dir = os.path.join("scraped_issues", repo_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{repo_path}_page_{page}.json") 
    
    try:
        with open(output_file, 'w') as f:
            json.dump(issue_list, f, indent=2)
        print(f"Saved {repo} page:{page} to {output_file}")
    except Exception as e:
        print(f"Error saving {repo} page:{page}: {e}")

def main():
    """Main function to scrape issues for multiple repositories."""
    
    github_token = os.getenv('GITHUB_AUTH_TOKEN')
    if not github_token:
        raise ValueError("GITHUB_AUTH_TOKEN not found in .env file")
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}"
    }
    
    for repo in repos:
        print(f"Scraping issues for {repo}...")
        fetch_issues(repo, headers)
    
    print("Scraping completed.")

if __name__ == "__main__":
    main()
