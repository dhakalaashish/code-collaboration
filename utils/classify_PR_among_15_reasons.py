import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep
from repos import repos  

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
    url = data["html_url"]

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
            
    return summary.strip(), has_locked_reason, merged, num_comments, num_review_comments, url

def generate_prompt(summary):
    return f"""
                You are tasked with classifying the reason why a Pull Request (PR) was closed based on the provided summary. You must strictly follow the predefined classification categories below and return an array of numbers corresponding to the relevant reasons. The classification should be based on clear indicators found in the summary, including PR discussions, review comments, labels, and metadata.

                Classification Categories:
                1. Superseded or replaced by another PR: A newer or alternative PR better addresses the issue, making this PR obsolete and unnecessary. Example indicators: "a newer PR fixes this issue," "this approach has been replaced by another PR."

                2. Erroneous or incorrect implementation: The PR contains incorrect logic, fundamental errors, or a misalignment with the intended solution. Example indicators: "incorrect approach," "logic is flawed," "does not work as expected."

                3. Inactive, abandoned, or withdrawn by author: The PR author explicitly withdraws it, stops responding, or does not address feedback requests. Example indicators: "author is unresponsive," "closing due to inactivity," "no updates from the author."

                4. Redundant, unnecessary, or low-value changes: The PR does not add substantial value or duplicates existing functionality. Example indicators: "already implemented elsewhere," "adds no real benefit," "redundant changes."

                5. Closed by design or intentionally: The project maintainers deliberately reject the PR based on strategic, architectural, or philosophical considerations. Example indicators: "this does not align with our roadmap," "intended behavior," "maintainers decided against this change."

                6. Miscommunication or misunderstanding: The PR was created based on an incorrect interpretation of project requirements or goals. Example indicators: "misunderstood the purpose," "not what we need," "this is not how this feature works."

                7. Outdated or no longer needed: The PR is no longer relevant due to project evolution, making the changes unnecessary. Example indicators: "this issue was fixed in a later version," "no longer applicable," "outdated implementation."

                8. Premature or not ready: The PR lacks completeness, maturity, or documentation, making it unfit for merging. Example indicators: "not enough detail," "missing essential components," "incomplete work."

                9. Build or test failures: The PR introduces issues that cause build or test failures and the author does not resolve them. Example indicators: "failing tests," "build breaks," "does not pass CI/CD checks."

                10. Unsafe, risky, or regressive changes: The PR introduces security vulnerabilities, instability, performance degradation, or breaks existing functionality. Example indicators: "introduces a security risk," "causes regressions," "negatively impacts performance."

                11. Incompatible, conflicts, or not aligning with the goals: The PR significantly conflicts with the existing architecture, project goals, or introduces breaking changes. Example indicators: "not compatible with our system," "conflicts with major features," "does not align with project objectives."

                12. Too large or out of scope: The PR is overly broad, complex, or introduces unnecessary scope creep. Example indicators: "this should be split into smaller PRs," "needs an RFC or community approval," "too ambitious for a single PR."

                13. Violation of contribution guidelines or licensing issues: The PR violates contribution rules, coding standards, documentation guidelines, or legal constraints. Example indicators: "violates our guidelines," "uses non-compliant code," "licensing issues."

                14. Community opposition: The PR receives strong pushback from maintainers or the community due to technical, strategic, or philosophical reasons. Example indicators: "community does not support this change," "pushback from maintainers," "not a direction we want to take."

                15. Other/Miscellaneous Reasons: The PR does not fit into any specific category but was still closed. Example indicators: "merged but still shows as closed," "closed for reasons not explicitly mentioned above."

                Instructions:
                Analyze the PR summary carefully.

                Identify the most relevant reasons from the list above.

                Return an array of numbers corresponding to the applicable classifications.

                If multiple reasons apply, include all relevant numbers.

                Do not generate any additional text, only output the array of numbers.

                Example Outputs:

                [1, 7] → The PR was superseded by another PR and is also outdated.

                [3] → The PR was abandoned by the author.

                [2, 9, 11] → The PR had incorrect logic, caused test failures, and conflicted with project goals.


            PR summary to analyze: "{summary}".
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

def extract_numbers(llm_response):
    extracted_numbers = [num for num in map(int, re.findall(r'\b\d+\b', llm_response)) if 1 <= num <= 15]
    return extracted_numbers if extracted_numbers else [15]  # Default to [15] if no valid numbers found

def main():
    for repo in repos:
        print(f"Started creating summary for {repo}")
        repo_path = repo.replace("/", "_")
        input_file = os.path.join(os.path.dirname(__file__), f"../scraped_data/{repo_path}.json")
        output_dir = os.path.join(os.path.dirname(__file__), "../ai_generated_15_reasons/repos")
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
            summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list, urls = zip(*summary_data)
        else:
            print(f"No unmerged PRs with comments for {repo}. Omitting this repo")
            continue

        print(f"Generating reasons for: {count} out of {total}")

        prompts = [generate_prompt(summary) for summary in summarys]

        print("Sending requests to Gemini API...")
        reasons = call_gemini(prompts)
        
        # Prepare the result data in a JSON-compatible format
        results = [
            {
                "url": url,
                "summary": summary,
                "hasLockedReason_merged_numComments_numReviewComments": [has_locked_reason, merged, num_comments, num_review_comments],
                "predicted_reason": extract_numbers(reason)
            }
            for summary, has_locked_reason, merged, num_comments, num_review_comments, url, reason in zip(summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list, urls, reasons)
        ]


        # Save results to JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {output_json}")

if __name__ == "__main__":
    main()