import matplotlib.pyplot as plt

# -----------------------------
# Global font settings
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# -----------------------------
# RFECV results (ordered from 157 → 10)
# -----------------------------
n_features = [157, 120, 80, 60, 42, 30, 10]
auroc_mean = [0.81, 0.84, 0.87, 0.885, 0.89, 0.86, 0.82]
auroc_std  = [0.03, 0.025, 0.02, 0.02, 0.02, 0.025, 0.03]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6.5, 4.2))

plt.errorbar(
    n_features,
    auroc_mean,
    yerr=auroc_std,
    fmt="o-",
    capsize=4,
    linewidth=1.8,
    markersize=6
)

plt.xlabel("Number of retained features", fontsize=12)
plt.ylabel("Cross-validated AUROC", fontsize=12)

# Highlight best point (42 features)
best_idx = auroc_mean.index(max(auroc_mean))
plt.scatter(
    n_features[best_idx],
    auroc_mean[best_idx],
    s=70,
    zorder=3
)

plt.annotate(
    f"{n_features[best_idx]} features\n"
    f"AUROC = {auroc_mean[best_idx]:.2f} ± {auroc_std[best_idx]:.2f}",
    (n_features[best_idx], auroc_mean[best_idx]),
    textcoords="offset points",
    xytext=(10, -22),
    ha="left",
    fontsize=10
)

# Axis formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

plt.tight_layout()
plt.show()
