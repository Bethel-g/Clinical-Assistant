import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set premium styling
plt.style.use('dark_background')
sns.set_theme(style='dark')

# Colors matching the application theme
theme_colors = {
    'primary': '#2dd4bf',  # Teal
    'secondary': '#38bdf8', # Sky Blue
    'accent': '#fbbf24',    # Amber
    'bg': '#0f172a',        # Deep Slate
    'panel': '#1e293b',     # Dark Slate
    'text': '#f8fafc',      # White-ish
    'muted': '#94a3b8'      # Slate gray
}

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=theme_colors['bg'])
fig.suptitle('EthioHealth-AI: Integrated Data & Model Performance Analytics', 
             color=theme_colors['text'], fontsize=18, fontweight='bold', y=0.98)

# 1. Dataset Composition Donut Chart
dataset_names = [
    'Ethiopian Hospital Dataset\n(50,000 records)', 
    'Kaggle Clinical CSV\n(4,693 records)', 
    'Kaggle Clinical XLSX\n(4,693 records)'
]
sizes = [50000, 4693, 4693]
colors = [theme_colors['primary'], theme_colors['secondary'], theme_colors['accent']]
explode = (0.05, 0.02, 0.02)

wedges, texts, autotexts = ax1.pie(
    sizes, explode=explode, labels=dataset_names, colors=colors,
    autopct='%1.1f%%', startangle=140, pctdistance=0.75,
    textprops=dict(color=theme_colors['text'], fontweight='bold')
)

# Draw circle in the middle to make it a donut chart
centre_circle = plt.Circle((0,0), 0.55, fc=theme_colors['bg'])
ax1.add_artist(centre_circle)
ax1.set_title('Unified Patient Database Composition\n(Total: 54,651 Records)', 
             color=theme_colors['muted'], fontsize=14, pad=20)

for autotext in autotexts:
    autotext.set_color(theme_colors['bg'])
    autotext.set_size(10)

# 2. Model Performance Metrics Bar Chart
metrics = ['Disease\nAccuracy', 'Disease\nF1-Score', 'Risk Level\nAccuracy', 'Risk Level\nF1-Score', 'Stay Prediction\nMAE (Days)']
values = [24.18, 22.97, 38.95, 39.01, 3.20]  # Stay MAE is in days, others in %

bars = ax2.bar(metrics, values, color=[theme_colors['secondary'], theme_colors['secondary'], theme_colors['primary'], theme_colors['primary'], theme_colors['accent']], width=0.6)

ax2.set_facecolor(theme_colors['panel'])
ax2.set_title('Evaluation Metrics (Test Set)', color=theme_colors['muted'], fontsize=14, pad=20)
ax2.set_ylim(0, 100)
ax2.grid(True, linestyle='--', alpha=0.1)

# Add values above bars
for bar in bars:
    height = bar.get_height()
    is_mae = bar.get_x() > 3.0  # Stay MAE bar
    label = f"{height:.2f} Days" if is_mae else f"{height:.2f}%"
    ax2.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', color=theme_colors['text'], fontweight='bold', fontsize=10)

# Customize axis labels
ax2.tick_params(colors=theme_colors['muted'], labelsize=10)
ax2.spines['bottom'].set_color(theme_colors['muted'])
ax2.spines['left'].set_color(theme_colors['muted'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
os.makedirs('assets', exist_ok=True)
plt.savefig('assets/performance_summary.png', dpi=300, bbox_inches='tight', facecolor=theme_colors['bg'])
print("Successfully generated assets/performance_summary.png")
