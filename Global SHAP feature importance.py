# -*- coding: utf-8 -*-
"""
From train_178.csv -> XGBoost -> SHAP -> Fig 6A (Top20 global importance) & Fig 6C (beeswarm)

Outputs:
  - Fig6A_SHAP_Global_Top20.png
  - Fig6C_SHAP_Beeswarm.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1) Load data ----------
TRAIN_PATH = "train_178.csv"   # 确保和你的文件同目录，或改成绝对路径
LABEL_COL = "label"

# 这些列不作为特征输入（与你前面数据结构一致）
NON_FEATURE_COLS = {
    "sample_id", "name_masked", "sex", "age_years", "acq_date", "batch", LABEL_COL
}

df = pd.read_csv(TRAIN_PATH)

# 选出特征列（仅数值列、且排除非特征字段）
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
if len(feature_cols) == 0:
    raise ValueError("未检测到特征列。请检查 NON_FEATURE_COLS 或 CSV 列名。")

X = df[feature_cols].to_numpy()
y = df[LABEL_COL].to_numpy().astype(int)

print(f"[INFO] Loaded {len(df)} samples, {len(feature_cols)} features from {TRAIN_PATH}")

# ---------- 2) Train XGBoost (Stage-1 TD vs HC) ----------
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError(
        "未能导入 xgboost。请先安装：pip install xgboost"
    ) from e

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)
print("[INFO] XGBoost trained on full training set.")

# ---------- 3) Compute SHAP values ----------
try:
    import shap
except Exception as e:
    raise ImportError(
        "未能导入 shap。请先安装：pip install shap"
    ) from e

# 为了可复现/兼容性更好：把 X 转成 DataFrame，保留列名
X_df = pd.DataFrame(X, columns=feature_cols)

explainer = shap.TreeExplainer(model)

# 不同版本 shap 对二分类返回格式不同：
# - 可能返回 (n_samples, n_features) 的 ndarray
# - 也可能返回 list/ndarray，包含正类/负类
shap_values = explainer.shap_values(X_df)

def get_positive_class_shap(shap_vals):
    """Return SHAP values for the positive class (TD=1) in binary classification."""
    # Case 1: list-like with two elements
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        return shap_vals[1]
    # Case 2: ndarray with shape (2, n_samples, n_features) or (n_samples, n_features)
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3 and shap_vals.shape[0] == 2:
            return shap_vals[1]
        if shap_vals.ndim == 2:
            return shap_vals
    # Fallback
    return shap_vals

shap_pos = get_positive_class_shap(shap_values)
shap_pos = np.array(shap_pos)  # ensure ndarray
if shap_pos.shape != X_df.shape:
    raise ValueError(f"SHAP shape {shap_pos.shape} does not match X shape {X_df.shape}")

print(f"[INFO] SHAP computed: shape = {shap_pos.shape} (samples x features)")

# ---------- 4) Fig 6A: Global importance (Top 20) ----------
mean_abs_shap = np.mean(np.abs(shap_pos), axis=0)  # per feature
imp = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

topN = 20
imp_top = imp.head(topN).iloc[::-1]  # reverse for horizontal bar (largest on top)

plt.figure(figsize=(7.2, 6.2))
plt.barh(imp_top["feature"], imp_top["mean_abs_shap"])
plt.xlabel("mean |SHAP|")
plt.title("Fig 6A: SHAP Global Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("Fig6A_SHAP_Global_Top20.png", dpi=300)
plt.show()

print("[INFO] Saved: Fig6A_SHAP_Global_Top20.png")

# ---------- 5) Fig 6C: SHAP beeswarm ----------
# shap.summary_plot 会自动按 mean(|SHAP|) 排序并绘制 beeswarm
# show=False 方便我们保存图片
plt.figure(figsize=(7.2, 6.2))
shap.summary_plot(
    shap_pos,
    X_df,
    plot_type="dot",
    max_display=20,   # 与 Fig6A 的 Top20 保持一致
    show=False
)
plt.title("Fig 6C: SHAP Beeswarm Plot (Top 20 Features)")
plt.tight_layout()
plt.savefig("Fig6C_SHAP_Beeswarm.png", dpi=300)
plt.show()

print("[INFO] Saved: Fig6C_SHAP_Beeswarm.png")

# ---------- 6) (Optional) Export Table-9-like values ----------
# 你如果需要直接生成“表9”的 mean|SHAP| 数值表：
table9 = imp.head(15).copy()
table9.to_csv("Table9_SHAP_Top15.csv", index=False)
print("[INFO] Saved: Table9_SHAP_Top15.csv")
