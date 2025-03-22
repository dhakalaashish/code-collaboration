import re


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
    Extracts all links in a given issue/PR and returns a dictionary of links and their types.
    """
    strings = []
    
    # Add issue body
    if issue.get('body'):
        strings.append(issue['body'])

    # Add comments
    if 'comments_url_body' in issue:
        comments = [comment['body'] for comment in issue['comments_url_body'] if comment.get('body')]
        strings.extend(comments)
    
    # Add review comments, commit messages if PR 
    if is_pr and 'pull_request_url_body' in issue:
        pr_body = issue['pull_request_url_body']

        # Safely extract review comments
        review_comments = [comment['body'] for comment in pr_body.get('review_comments_url_body', []) if comment.get('body')]
        strings.extend(review_comments)

        # Add commit message
        commit_message = pr_body.get('commit_message')
        if commit_message:
            strings.append(commit_message)
    
    return extract_links_from_text(strings) if strings else []
