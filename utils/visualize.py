import json
import matplotlib.pyplot as plt

# Load JSON data from the file
input_file = "summary_with_predicted_reason/reason_counts.json"
with open(input_file, "r", encoding="utf-8") as f:
    reason_counts = json.load(f)

# Extract keys and values for plotting
reasons = list(reason_counts.keys())
counts = list(reason_counts.values())

# Create a Bar Graph
plt.figure(figsize=(10, 6))
plt.barh(reasons, counts, color='skyblue')
plt.xlabel('Count')
plt.ylabel('Reasons')
plt.title('Reason Counts (Bar Graph)')
plt.tight_layout()
plt.savefig('reason_counts_bar_graph.png')  # Save the bar graph as a PNG file
plt.clf()  # Clear the current figure for the next plot

# Create a Pie Chart
# Create a Pie Chart
plt.figure(figsize=(12, 12))  # Increase the size of the figure
plt.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Reason Counts (Pie Chart)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()  # Ensure everything fits without cutting off
plt.savefig('reason_counts_pie_chart.png')  # Save the pie chart as a PNG file

print("Bar graph and pie chart saved as 'reason_counts_bar_graph.png' and 'reason_counts_pie_chart.png'")
