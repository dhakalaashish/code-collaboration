import os
import json
import time
import requests
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_links_from_text(string_list):
    """
    Function to extract instances of URLs or issue/PR numbers from a list of strings
    """
    url_pattern = r'https?://\S+'
    issue_references_pattern = r'(?:fixes|mentioned in|resolves|closes) #?(\d+)'
    
    urls, issue_references = [], []
    for string in string_list:
        urls.extend(re.findall(url_pattern, string))
        issue_references.extend(re.findall(issue_references_pattern, string, re.IGNORECASE))
    
    links_to = []
    for url in urls:
        type = 'url'
        if 'pull' in url:
            type = 'pull_url'
        elif 'issue' in url:
            type = 'issue_url'
        links_to.append({'link':url, 'type':type})
    
    for issue_reference in issue_references:
        links_to.append({'link':f'#{issue_reference}', 'type':'Issue/PR number'})

    return links_to


def extract_all_links(issue, is_pr):
    """
    extracts all links in a given issue/PR and returns a dictionary of links anf their types
    """
    strings = []
    # add issue body
    strings.append(issue['body'])

    # add comments
    comments = [comment['body'] for comment in issue['comments_url_body']]
    strings.extend(comments)
    
    # add review comments, commit messages if PR 
    if is_pr:
        review_comments = [comment['body'] for comment in issue['pull_request_url_body']['review_comments_url_body']]
        strings.extend(review_comments)

        commit_message = issue['pull_request_url_body']['commit_message']
        strings.append(commit_message)
    
    links_to = extract_links_from_text(strings)
    
    return links_to

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
    """Fetch all closed issues for a given repo."""
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "closed", "per_page": 100, "page": 1}
    all_issues = []
    
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
            all_issues.extend(data)
            params["page"] += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issues for {repo}: {e}")
            break
    
    return all_issues

def save_issues(repo, issues):
    """Save issues to a JSON file in the correct format."""
    repo_path = repo.replace("/", "_")
    output_dir = os.path.join("scraped_issues", repo_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{repo_path}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(issues, f, indent=2)
        print(f"Saved issues for {repo} to {output_file}")
    except Exception as e:
        print(f"Error saving issues for {repo}: {e}")

def main():
    """Main function to scrape issues for multiple repositories."""
    repos = ["jax-ml/jax"]
    
    github_token = os.getenv('GITHUB_AUTH_TOKEN')
    if not github_token:
        raise ValueError("GITHUB_AUTH_TOKEN not found in .env file")
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}"
    }
    
    for repo in repos:
        print(f"Scraping issues for {repo}...")
        issues = fetch_issues(repo, headers)
        save_issues(repo, issues)
    
    print("Scraping completed.")

if __name__ == "__main__":
    main()
