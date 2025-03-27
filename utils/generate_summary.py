import os
import json
import csv
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep
from repos import repos  # Import repo list from the original script

# Load JSON data from the file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_to_summary(data):
    has_locked_reason = False
    merged = False
    num_comments = 0
    num_review_comments = 0

    summary = (
        f"Pull Request '{data['number']}' titled '{data['title']}' was authored by a {data['user']['type']}, who is associated as a {data['author_association']}. "
        f"\nIt was created at {data['created_at']}, and was closed at {data['closed_at']} {('by a ' + data['closed_by']['type']) if data.get('closed_by') else 'N/A'}.\n"
    )

    if 'labels' in data:
        labels = data['labels']
        if len(labels) > 0:
            labels_text = 'The PR has labels: '
            for i in range(len(labels)):
                if i != len(labels)-1:
                    labels_text += f"{labels[i]['name']} - {labels[i]['description']}, "
                else:
                    labels_text += f"{labels[i]['name']} - {labels[i]['description']}. "
            summary += labels_text + '\n'

    if 'locked' in data and data["locked"] and 'active_lock_reason' in data and data['active_lock_reason']:
        has_locked_reason = True
        summary += f"PR was locked because of {data['active_lock_reason']}.\n"

    if data['body']:
        summary += f"It has a body of '{data['body']}'\n"

    if 'comments_url_body' in data and data['comments_url_body']:
        comments = data['comments_url_body']
        num_comments = len(comments)
        summary += f"PR has comments:\n"
        for i in range(num_comments):
            summary += f"'{comments[i]['body']}' by a {comments[i]['author_association']} of type {comments[i]['user']['type']} on {comments[i]['created_at']}\n"
        summary += '\n'

    if 'pull_request' in data:
        if data['pull_request']['merged_at']:
            merged = True
            summary += f"It was merged at {data['pull_request']['merged_at']} by a {data['pull_request_url_body']['merged_by']['type']}.\n "

        # if data['pull_request']['patch_url_body']:
        #     summary += f"The PR includes the following patch:\n{data['pull_request']['patch_url_body']}.\n"

        if data['pull_request_url_body']['review_comments_url_body']:
            review_comments = data['pull_request_url_body']['review_comments_url_body']
            num_review_comments = len(review_comments)
            summary += f"PR has review comments:\n"
            for review in review_comments:
                if review and 'user' in review and review['user']:  # Ensure review and user exist
                    user_type = review['user'].get('type', 'Unknown')
                    summary += f"'{review.get('body', 'No body available')}' by a {review.get('author_association', 'Unknown')} of type {user_type} on {review.get('created_at', 'Unknown date')}\n"
            summary += '\n'
            
    return summary.strip(), has_locked_reason, merged, num_comments, num_review_comments

def main():
    for repo in repos:
        print(f"Started creating summary for {repo}")
        repo_path = repo.replace("/", "_")
        input_file = os.path.join(os.path.dirname(__file__), f"../scraped_data/{repo_path}.json")
        output_dir = os.path.join(os.path.dirname(__file__), "../summaries_unmerged_commented_pr_only")
        output_json = os.path.join(output_dir, f"{repo_path}.json")

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found.")
            return

        data = load_json(input_file)
        total = 0
        count = 0
        if isinstance(data, list):  # If JSON is a list of PRs, process each one
            summary_data = []
            for entry in data:
                total += 1
                num_comments = len(entry.get('comments_url_body', []))
                if 'pull_request' in entry and entry['pull_request']:
                    pull_request_data = entry.get('pull_request_url_body', {})
                    review_comments = pull_request_data.get('review_comments_url_body', [])

                    # Ensure review_comments is a list before calling len()
                    if review_comments is None:
                        num_review_comments = 0
                    else:
                        num_review_comments = len(review_comments)

                    pull_request = entry['pull_request']
                    if not pull_request.get('merged_at') and num_comments > 0 and num_review_comments > 0:
                        count += 1
                        summary_data.append(json_to_summary(entry))

        else:  # If JSON contains only a single PR, process it directly
            if 'pull_request' in data and data['pull_request'] and not data['pull_request'].get('merged_at'):
                summary_data = [json_to_summary(data)]
            else:
                summary_data = []  # or handle the case where no matching PR is found

        if summary_data:
            summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list = zip(*summary_data)
        else:
            print(f"No unmerged PRs with comments for {repo}. Omitting this repo")
            continue

        # Prepare the result data in a JSON-compatible format
        results = [
            {
                "summary": summary,
                "hasLockedReason_merged_numComments_numReviewComments": [has_locked_reason, merged, num_comments, num_review_comments]
            }
            for summary, has_locked_reason, merged, num_comments, num_review_comments in zip(summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list)
        ]


        # Save results to JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"{repo} has {count} unmerged PR with comments out of {total}")
        print(f"Results saved to {output_json}")


if __name__ == "__main__":
    main()