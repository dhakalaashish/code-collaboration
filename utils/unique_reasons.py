import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Load the JSON data from just_reasons.json
with open("ai_accuracy/just_reasons.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract all human reasons
human_reasons = [item["human_reason"] for item in data]

# Generate a prompt for the API
prompt = f"""
I have a list of reasons for pull request closures. Your task is to analyze them and provide a set of unique closure reasons. Here is the list of reasons:

{json.dumps(human_reasons, indent=2)}

Please return a concise list of distinct reasons that generalize the closures. Each unique reason must be 5 words maximum.
"""

# Call the Gemini API
response = model.generate_content(prompt)

# Extract the unique reasons from the response
unique_reasons = response.text.strip()

# Save the unique reasons to a JSON file
output_file = "ai_accuracy/unique_reasons.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"unique_reasons": unique_reasons}, f, indent=4)

print(f"Unique reasons saved to {output_file}!")