import os
import json
import numpy as np
import random
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from utils.generate_reasons import json_to_paragraph, load_json

# Load dataset
input_file = os.path.join(os.path.dirname(__file__), "../scraped_data/unmerged_prs_jax-ml_jax.json")
output_file = os.path.join(os.path.dirname(__file__), "clustered_issues.json")

if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found.")

data = load_json(input_file)

# Convert all JSON entries to paragraphs
paragraphs = [json_to_paragraph(entry) for entry in data]

# Generate embeddings using BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(paragraphs)

# Apply K-Means Clustering
num_clusters = 30  # Adjust based on data size
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Organize clustered issues
clustered_issues = {i: [] for i in range(num_clusters)}
for i, label in enumerate(labels):
    clustered_issues[label].append(paragraphs[i])

# Save clustered issues to a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(clustered_issues, f, indent=4)

print(f"Clustered issues saved to {output_file}")

# Randomly select one issue per cluster
random.seed(42)  # Ensure reproducibility
selected_issues = {cluster: random.choice(issues) for cluster, issues in clustered_issues.items() if issues}

# Save selected issues to a separate file
selected_output_file = os.path.join(os.path.dirname(__file__), "selected_issues.json")
with open(selected_output_file, "w", encoding="utf-8") as f:
    json.dump(selected_issues, f, indent=4)

print(f"Randomly selected issues saved to {selected_output_file}")

# Print selected issues for verification
for cluster, issue in selected_issues.items():
    print(f"\nCluster {cluster} (Selected Issue):")
    print(issue[:300], "...")  # Truncate for readability