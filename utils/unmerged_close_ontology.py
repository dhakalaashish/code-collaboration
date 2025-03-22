import json
import os

print("Started...")

# Load the JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Extract relevant user details
def extract_user_info(user):
    if not user:
        return None
    return {
        "login": user.get("login"),
        "id": user.get("id"),
        "type": user.get("type"),
        "site_admin": user.get("site_admin"),
    }

# Extract and process review comments from pull_request_url_body
def extract_review_comments(pr_body):
    if not pr_body or "review_comments_url_body" not in pr_body:
        return None

    comments = [
        {
            "url": comment.get("url"),
            "pull_request_review_id": comment.get("pull_request_review_id"),
            "id": comment.get("id"),
            "node_id": comment.get("node_id"),
            "diff_hunk": comment.get("diff_hunk"),
            "path": comment.get("path"),
            "commit_id": comment.get("commit_id"),
            "original_commit_id": comment.get("original_commit_id"),
            "user": extract_user_info(comment.get("user")),  # Condense user data
            "body": comment.get("body"),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "author_association": comment.get("author_association"),
            "reactions": comment.get("reactions"),
            "start_line": comment.get("start_line"),
            "original_start_line": comment.get("original_start_line"),
            "start_side": comment.get("start_side"),
            "line": comment.get("line"),
            "original_line": comment.get("original_line"),
            "side": comment.get("side"),
            "original_position": comment.get("original_position"),
            "position": comment.get("position"),
            "subject_type": comment.get("subject_type"),
        }
        for comment in pr_body.get("review_comments_url_body", [])
    ]

    return comments if comments else None  # Return None if there are no comments

# Filter unmerged PRs and retain only the relevant attributes
def filter_unmerged_prs(data):
    filtered_prs = []
    for pr in data:
        if pr.get("pull_request") and pr.get("pull_request").get("merged_at") is None:  # Not merged
            filtered_pr = {
                "id": pr.get("id"),
                "number": pr.get("number"),
                "title": pr.get("title"),
                "user": extract_user_info(pr.get("user")),  # Extract only needed fields
                "state": pr.get("state"),
                "closed_by": extract_user_info(pr.get("closed_by")),  # Extract only needed fields
                "state_reason": pr.get("state_reason"),
                "comments_url_body": pr.get("comments_url_body"),
                "review_comments_url_body": extract_review_comments(pr.get("pull_request_url_body")),  # Extract from nested structure
            }
            filtered_prs.append(filtered_pr)
    return filtered_prs

# Save filtered data
def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Main function
def main():
    input_file = os.path.join(os.path.dirname(__file__), "../scraped_data/jax-ml_jax.json")
    output_file = os.path.join(os.path.dirname(__file__), "../scraped_data/unmerged_prs_jax-ml_jax.json")

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    data = load_json(input_file)
    filtered_data = filter_unmerged_prs(data)
    save_json(filtered_data, output_file)
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    main()