import os
import json
import csv
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# models = genai.list_models()
# for m in models:
#     print(m.name, m.supported_generation_methods)

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

    if data['comments_url_body']:
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

        if data['pull_request']['patch_url_body']:
            summary += f"The PR includes the following patch:\n{data['pull_request']['patch_url_body']}.\n"

        if data['pull_request_url_body']['review_comments_url_body']:
            review_comments = data['pull_request_url_body']['review_comments_url_body']
            num_review_comments = len(review_comments)
            summary += f"PR has review comments:\n"
            for i in range(num_review_comments):
                summary += f"'{review_comments[i]['body']}' by a {review_comments[i]['author_association']} of type {review_comments[i]['user']['type']} on {review_comments[i]['created_at']}\n"
            summary += '\n'
            
    return summary.strip(), has_locked_reason, merged, num_comments, num_review_comments

def generate_prompt(summary):
    return f"""
                You are an AI assistant analyzing GitHub pull requests. Based on the following PR details, determine the most likely reason why it was closed without being merged. 

                {summary}

                Provide a concise and clear reason for closure.
            """

def call_gemini(prompts):
    results = []
    batch_size = 10

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        try:
            responses = [model.generate_content(p) for p in batch]
            reasons = [resp.text.strip() if resp.text else "No response" for resp in responses]
        except Exception as e:
            print(f"Error during API call: {e}")
            reasons = ["API Error"] * len(batch)

        results.extend(reasons)
        sleep(2)  # Prevent hitting rate limits

    return results

def main():
    # input_file = os.path.join(os.path.dirname(__file__), "../example_data/cleaned_extended_issue.json")
    input_file = os.path.join(os.path.dirname(__file__), "../scraped_data/jax-ml_jax.json")
    output_json = os.path.join(os.path.dirname(__file__), "pr_closure_reasons.json")

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
            num_comments = len(entry['comments_url_body'])
            if 'pull_request' in entry and entry['pull_request']:
                num_review_comments = len(entry['pull_request_url_body']['review_comments_url_body'])
                pull_request = entry['pull_request']
                if not pull_request.get('merged_at') and num_comments > 0 and num_review_comments > 0:
                    count += 1
                    summary_data.append(json_to_summary(entry))

    else:  # If JSON contains only a single PR, process it directly
        if 'pull_request' in data and data['pull_request'] and not data['pull_request'].get('merged_at'):
            summary_data = [json_to_summary(data)]
        else:
            summary_data = []  # or handle the case where no matching PR is found

    summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list = zip(*summary_data)

    print(f"Generating reasons for: {count} out of {total}")

    prompts = [generate_prompt(summary) for summary in summarys]

    print("Sending requests to Gemini API...")
    reasons = call_gemini(prompts)
    
    # Prepare the result data in a JSON-compatible format
    results = [
        {
            "summary": summary,
            "has_locked_reason": has_locked_reason,
            "merged": merged,
            "num_comments": num_comments,
            "num_review_comments": num_review_comments,
            "reason_for_closure": reason
        }
        for summary, has_locked_reason, merged, num_comments, num_review_comments, reason in zip(summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list, reasons)
    ]

    # Save results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    main()