import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from time import sleep

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Load JSON data
input_file = "ai_accuracy/just_reasons.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def get_similarity_score(human_reason, predicted_reason):
    prompt = f"""
    You are an AI evaluating the similarity between two explanations for why a pull request was closed.
    
    **Instructions:**
    - Given the human-provided reason and the AI-predicted reason, assign a similarity score between 1 and 10.
    - A score of 10 means the reasons are nearly identical.
    - A score of 1 means the reasons are completely different.
    - Consider semantic meaning rather than exact wording.
    
    **Human Reason:** {human_reason}
    **Predicted Reason:** {predicted_reason}
    
    **Provide only a number between 1 and 10 as the response.**
    """
    
    retries = 3  # Retry mechanism in case of API failure
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            score = int(response.text.strip())  # Ensure response is a number
            if 1 <= score <= 10:
                return score
        except (ValueError, TypeError):
            print(f"Invalid response from API. Retrying... ({attempt + 1}/{retries})")
        sleep(2)  # Wait before retrying
    
    return None  # Return None for failed cases

# Compute similarity scores
total_score = 0
total_total = 0
num_entries = len(data)
scores = []

for item in data:
    human_reason = item["human_reason"]
    predicted_reason = item["predicted_reason"]
    score = get_similarity_score(human_reason, predicted_reason)
    if score is not None:  # Ignore failed cases
        scores.append({
            "human_reason": human_reason,
            "predicted_reason": predicted_reason,
            "score": score
        })
        total_score += score
        total_total += 10

# Compute final accuracy, avoiding division by zero
accuracy = (total_score / total_total) * 100 if total_total > 0 else 0

# Save results
output_file = "ai_accuracy/similarity_scores.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=4)

print(f"Similarity scores saved to {output_file}!")
print(f"Overall Accuracy: {accuracy:.2f}%")
