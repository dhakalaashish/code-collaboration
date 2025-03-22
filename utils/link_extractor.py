import re


def extract_links_from_text(string_list):
    """
    Function to extract instances of URLs or issue/PR numbers from a list of strings
    """
    url_pattern = r'https?://\S+'
    issue_references_pattern = r'(?:fixes|mentioned in|resolves|closes)[\s#]*(\d+)'
    
    urls, issue_references = [], []
    for string in string_list:
        urls.extend(re.findall(url_pattern, string))
        issue_references.extend(re.findall(issue_references_pattern, string, re.IGNORECASE))
    
    links_to = []
    for url in urls:
        url = url.rstrip('.,!?')  # Remove trailing punctuation
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
    if issue.get('body'):
        strings.append(issue['body'])

    # add comments
    if issue.get('comments_url_body'):
        comments = [comment.get('body') for comment in issue.get('comments_url_body', []) if comment.get('body')]
        strings.extend(comments)
    
    # add review comments and commit messages if PR
    if is_pr and issue.get('pull_request_url_body'):
        pr_data = issue['pull_request_url_body']
        
        if pr_data.get('review_comments_url_body'):
            review_comments = [comment.get('body') for comment in pr_data.get('review_comments_url_body', []) if comment.get('body')]
            strings.extend(review_comments)

        if pr_data.get('commit_message'):
            strings.append(pr_data['commit_message'])
    
    return extract_links_from_text(strings) if strings else []
