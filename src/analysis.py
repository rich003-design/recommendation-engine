# analyze_data.py
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis import analyze_engagement, plot_metrics

# Load cleaned data
df = pd.read_csv('data/processed/cleaned_ga4_data.csv')

# Perform analysis
results = analyze_engagement(df)
print(results)

# Create visualizations
plot_metrics(df)
plt.savefig('reports/engagement_metrics.png')