import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from collections import defaultdict
from time import sleep
from repos import repos  

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Define the 16 unique reasons + Fallback reason 17
unique_reasons = {
    1: "Superseded/Replaced by other PR",
    2: "Erroneous/Incorrect implementation",
    3: "Inactivity/Lack of response",
    4: "Redundant/Unnecessary changes",
    5: "Withdrawn/Abandoned by author",
    6: "Not useful/Low value",
    7: "Closed by design/Intentionally",
    8: "Other/Miscellaneous reasons",
    9: "Miscommunication/Misunderstanding",
    10: "Outdated/No longer needed",
    11: "Premature/Not ready yet",
    12: "Build/Test failed",
    13: "Unsafe/Risky changes",
    14: "Incompatible/Conflicts with existing",
    15: "Too large/Needs separate proposal",
    16: "Merged PR/ Incorrectly labeled as unmerged",
    17: "API Failure - Unable to Determine",
}


# Function to get reason number from Gemini API
def get_reason_number(summary):
    prompt = f"""
    You are an AI agent tasked with categorizing pull request closures based on the following reasons:

    1. Superseded/Replaced by other PR
    2. Erroneous/Incorrect implementation
    3. Inactivity/Lack of response
    4. Redundant/Unnecessary changes
    5. Withdrawn/Abandoned by author
    6. Not useful/Low value
    7. Closed by design/Intentionally
    8. Other/Miscellaneous reasons
    9. Miscommunication/Misunderstanding
    10. Outdated/No longer needed
    11. Premature/Not ready yet
    12. Build/Test failed
    13. Unsafe/Risky changes
    14. Incompatible/Conflicts with existing
    15. Too large/Needs separate proposal
    16. Merged PR/ Incorrectly labeled as unmerged

    **Instructions:** Given the PR summary below, return ONLY a number from 1 to 16 that best represents the reason for closure. Do not include explanations or any other text.

    PR Summary:
    {summary}
    """

    retries = 3  # Retry mechanism in case of API failure
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            reason_number = int(response.text.strip())  # Ensure response is a number
            if reason_number in unique_reasons:
                return reason_number
        except (ValueError, TypeError):
            print(f"Invalid response from API. Retrying... ({attempt + 1}/{retries})")
        sleep(2)  # Wait before retrying

    return 17  # Default to "API Failure - Unable to Determine" if API fails

# Dictionary to count occurrences of each reason
reason_counts = defaultdict(int)
reason_counts[17] = 0  # Ensure reason 17 is tracked
results = []

for repo in repos:
    # Load summaries from just_reasons.json
    print(f"Started creating summary for {repo}")
    repo_path = repo.replace("/", "_")
    input_file = f"summary_with_predicted_reason/repos/{repo_path}.json"
    output_file = f"summary_with_predicted_reason/fixed_option_reasons.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each summary
    for item in data:
        summary = item["summary"]
        predicted_reason_number = get_reason_number(summary)
        reason_counts[predicted_reason_number] += 1

        results.append({
            "repository": repo,
            "summary": summary,
            "predicted_reason": unique_reasons[predicted_reason_number],
        })

# Save the results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# Save reason count statistics
with open("summary_with_predicted_reason/reason_counts.json", "w", encoding="utf-8") as f:
    json.dump(reason_counts, f, indent=4)

print(f"Predicted reasons saved to {output_file}!")
print(f"Reason counts saved to summary_with_predicted_reason/reason_counts.json!")
