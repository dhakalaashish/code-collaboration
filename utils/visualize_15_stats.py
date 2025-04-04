import json
import matplotlib.pyplot as plt

# Load JSON data from the updated file path
input_file = "ai_generated_15_reasons/analysis/reason_counts.json"
with open(input_file, "r", encoding="utf-8") as f:
    reason_counts = json.load(f)

# Extract keys and values for plotting
reasons = list(reason_counts.keys())
counts = list(reason_counts.values())

# Convert numeric reason keys to sorted labels
reasons = [int(r) for r in reasons]  # Convert to int for sorting
sorted_pairs = sorted(zip(reasons, counts))  # Sort by reason number
reasons, counts = zip(*sorted_pairs)  # Unzip sorted pairs

# Convert reasons back to strings for labeling
reasons = [str(r) for r in reasons]

# Create a Bar Graph
plt.figure(figsize=(10, 6))
plt.barh(reasons, counts, color='skyblue')
plt.xlabel('Count')
plt.ylabel('Reasons')
plt.title('Reason Counts (Bar Graph)')
plt.tight_layout()
plt.savefig('ai_generated_15_reasons/analysis/reason_counts_bar_graph.png')  # Save the bar graph
plt.clf()  # Clear the figure for the next plot

# Create a Pie Chart
plt.figure(figsize=(10, 10))
plt.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Reason Counts (Pie Chart)')
plt.axis('equal')  # Ensures the pie is drawn as a circle
plt.tight_layout()
plt.savefig('ai_generated_15_reasons/analysis/reason_counts_pie_chart.png')  # Save the pie chart

print("Bar graph and pie chart saved in 'ai_generated_15_reasons/analysis/'")