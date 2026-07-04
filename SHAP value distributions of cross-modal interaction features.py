import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global style
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
np.random.seed(42)

# -----------------------------
# Interaction names (use EEG feature names)
# -----------------------------
interaction_names = [
    "Frontal β-band power × 25-OH vitamin D",
    "Central γ-band power × 25-OH vitamin D",
    "Inferior frontal β-PSE × 25-OH vitamin D",
    "Frontal non-stationarity index × 25-OH vitamin D",
    "Temporal time–frequency entropy × 25-OH vitamin D",
]

# -----------------------------
# Synergy gain (your confirmed values)
# -----------------------------
synergy_gain = [0.016, 0.013, 0.011, 0.007, 0.006]

# -----------------------------
# Simulate SHAP values (for plotting only)
# Replace this with real SHAP interaction values if you have them.
# -----------------------------
n_samples = 180
base_scale = 0.018
shap_data = [
    np.random.normal(loc=0, scale=max(0.006, base_scale - i * 0.002), size=n_samples)
    for i in range(len(interaction_names))
]

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 5))

for i, values in enumerate(shap_data):
    # add jitter on y-axis to avoid striping
    y = np.random.normal(loc=i, scale=0.08, size=len(values))
    ax.scatter(
        values,
        y,
        s=14,
        alpha=0.7,
        edgecolors="none"
    )

# Zero reference line
ax.axvline(0, linewidth=1.4)

# -----------------------------
# Axes formatting
# -----------------------------
ax.set_yticks(range(len(interaction_names)))
ax.set_yticklabels(interaction_names, fontsize=12)

ax.set_xlabel("SHAP value", fontsize=16)
ax.set_ylabel("Interaction feature", fontsize=16)

ax.tick_params(axis="x", labelsize=12)
ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)

# -----------------------------
# Right axis: Synergy Gain
# -----------------------------
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(interaction_names)))
ax2.set_yticklabels([f"{sg:.3f}" for sg in synergy_gain], fontsize=12)
ax2.set_ylabel("Synergy gain", fontsize=16)

# -----------------------------
# Layout & show
# -----------------------------
plt.tight_layout()
plt.show()
