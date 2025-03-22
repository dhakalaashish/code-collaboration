import os
import json
import random
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage  # Import HumanMessage

# Load environment variables
load_dotenv()

openai = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["title", "description", "comments"],
    template="""
    You are an AI assistant analyzing GitHub pull requests. Based on the following PR details, determine the most likely reason why it was closed without being merged. 

    Title: {title}
    Description: {description}
    Comments: {comments}

    Provide a concise and clear reason for closure.
    """
)

# Function to generate a reason for PR closure using OpenAI
def generate_reason(pr_data):
    prompt = prompt_template.format(
        title=pr_data.get("title", "N/A"),
        description=pr_data.get("body", "No description provided."),
        comments=pr_data.get("comments", "No comments available.")
    )

    # Fix: Use HumanMessage for correct input format
    response = openai.invoke([HumanMessage(content=prompt)])
    return response.content  # Extract text response

# Function to parse PR details from a string
def parse_pr_details(pr_string):
    """
    Extracts title, body, and comments from the PR string.
    """
    title_match = re.search(r"Pull Request \d+ titled '(.+?)'", pr_string)
    title = title_match.group(1) if title_match else "N/A"

    body_match = re.search(r"Body: (.+?)(?:\n\n|$)", pr_string, re.DOTALL)
    body = body_match.group(1).strip() if body_match else "No description available"

    comments_match = re.findall(r"Comments:\n- (.+)", pr_string, re.DOTALL)
    comments = "\n".join(comments_match) if comments_match else "No comments available."

    return {
        "title": title,
        "body": body,
        "comments": comments
    }

# Load clustered issues
clustered_issues_file = os.path.join(os.path.dirname(__file__), "clustered_issues.json")
reasons_output_file = "generated_reasons.json"

if not os.path.exists(clustered_issues_file):
    print(f"Error: File '{clustered_issues_file}' not found.")
else:
    with open(clustered_issues_file, "r", encoding="utf-8") as f:
        clustered_issues = json.load(f)

    # Dictionary to store generated reasons
    generated_reasons = {}

    # Analyze a random PR from each cluster
    for cluster, issues in clustered_issues.items():
        if issues:
            selected_pr = random.choice(issues)  # Selected PR is a string
            parsed_pr = parse_pr_details(selected_pr)  # Convert it to a dictionary
            
            reason = generate_reason(parsed_pr)  # Pass the structured dictionary
            
            # Store the reason in the dictionary
            generated_reasons[cluster] = {
                "selected_pr_title": parsed_pr["title"],
                "reason_for_closure": reason
            }

    # Save the generated reasons to a JSON file
    with open(reasons_output_file, "w", encoding="utf-8") as f:
        json.dump(generated_reasons, f, indent=4)

    print(f"Generated reasons saved to {reasons_output_file}")