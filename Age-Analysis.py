# -*- coding: utf-8 -*-
"""
ONE model on full filtered cohort; SHAP overall + age-stratified (combined into ONE figure)
+ Export age-stratified Top-5 features (by mean|SHAP|) to Excel/CSV (Table 12)

pip install pandas numpy scikit-learn xgboost shap matplotlib openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
import shap


CSV_PATH = "master_178.csv"
OUT_DIR = "shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

AGE_MIN, AGE_MAX = 3.0, 12.9
TOP_N = 20          # for beeswarm display
TOP_K_TABLE = 5     # for Table 12
RANDOM_STATE = 42


def safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def assign_age_group(a: float) -> str:
    if 3.0 <= a <= 6.9:
        return "3.0–6.9"
    if 7.0 <= a <= 9.9:
        return "7.0–9.9"
    if 10.0 <= a <= 12.9:
        return "10.0–12.9"
    return "OUT"


def get_feature_names(preprocessor: ColumnTransformer):
    """Get feature names from a fitted ColumnTransformer."""
    out = []
    # numeric pipeline
    num_features = preprocessor.transformers_[0][2]
    out.extend(list(num_features))
    # categorical pipeline
    cat_features = preprocessor.transformers_[1][2]
    if len(cat_features) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        try:
            cat_names = list(ohe.get_feature_names_out(cat_features))
        except AttributeError:
            cat_names = list(ohe.get_feature_names(cat_features))
        out.extend(cat_names)
    return np.array(out)


def save_beeswarm(shap_values, X_transformed, feature_names, out_path, title, max_display=20):
    """Save a standard SHAP beeswarm plot."""
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_topk_by_age_to_excel(df, shap_values_pos, feature_names, out_dir, top_k=5,
                               age_groups=("3.0–6.9", "7.0–9.9", "10.0–12.9"),
                               excel_name="Table12_AgeStratified_Top5_SHAP.xlsx",
                               csv_name="Table12_AgeStratified_Top5_SHAP.csv"):
    """
    Use the same SHAP matrices used for plotting:
      shap_g = shap_values_pos[pos, :]
    Compute mean|SHAP| within each age group and export Top-K.
    """
    rows = []
    for g in age_groups:
        pos = np.where(df["age_group"].values == g)[0]
        if len(pos) == 0:
            continue

        shap_g = shap_values_pos[pos, :]              # SAME source as beeswarm
        mean_abs = np.abs(shap_g).mean(axis=0)        # mean |SHAP| within age group
        top_idx = np.argsort(mean_abs)[-top_k:][::-1] # Top-K indices, descending

        for rank, fi in enumerate(top_idx, start=1):
            rows.append({
                "Age group": g,
                "Rank": rank,
                "Feature": str(feature_names[fi]),
                "mean|SHAP|": float(mean_abs[fi]),
                "n (age group)": int(len(pos))
            })

    out_df = pd.DataFrame(rows)

    xlsx_path = os.path.join(out_dir, excel_name)
    csv_path = os.path.join(out_dir, csv_name)

    out_df.to_excel(xlsx_path, index=False)
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n[Table12] Saved:")
    print("XLSX:", os.path.abspath(xlsx_path))
    print("CSV :", os.path.abspath(csv_path))

    # Console preview for quick copy-paste
    for g in age_groups:
        sub = out_df[out_df["Age group"] == g].sort_values("Rank")
        if len(sub) == 0:
            continue
        print(f"\n=== Age {g} | Top {top_k} (by mean|SHAP|) | n={sub['n (age group)'].iloc[0]} ===")
        print(sub[["Rank", "Feature", "mean|SHAP|"]].to_string(index=False))

    return out_df


# -----------------------------
# Load & filter
# -----------------------------
df = pd.read_csv(CSV_PATH)

required_cols = {"age_years", "label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV缺少必要列: {missing}. 当前列名示例: {list(df.columns)[:30]}")

df = df[(df["age_years"] >= AGE_MIN) & (df["age_years"] <= AGE_MAX)].copy()
df["age_group"] = df["age_years"].apply(assign_age_group)

print("Filtered n =", len(df))
print("Age range:", df["age_years"].min(), "to", df["age_years"].max())
print("Label counts:\n", df["label"].value_counts())
print("Age group counts:\n", df["age_group"].value_counts())

# -----------------------------
# Build X/y
# -----------------------------
target_col = "label"
DROP_COLS = ["sample_id", "acq_date", "name_masked", "age_group"]
drop_exist = [c for c in DROP_COLS if c in df.columns]

X = df.drop(columns=drop_exist + [target_col]).copy()
y = df[target_col].astype(int).values

# Robust column typing:
numeric_cols = []
categorical_cols = []

for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

print("Detected numeric cols:", len(numeric_cols))
print("Detected categorical cols:", categorical_cols)

# Coerce numeric cols to float
for c in numeric_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# -----------------------------
# Preprocess + ONE model
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", safe_ohe()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

model = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)

pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
pipe.fit(X, y)

# Transform for SHAP
X_trans = pipe.named_steps["prep"].transform(X)
feature_names = get_feature_names(pipe.named_steps["prep"])

# -----------------------------
# SHAP
# -----------------------------
clf = pipe.named_steps["clf"]
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_trans)
shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

# -----------------------------
# Figure 8 overall
# -----------------------------
overall_path = os.path.join(OUT_DIR, "Figure8_overall_beeswarm.png")
save_beeswarm(
    shap_values_pos,
    X_trans,
    feature_names,
    overall_path,
    title="Figure 8. SHAP beeswarm (All samples, age 3.0–12.9)",
    max_display=TOP_N
)
print("Saved:", overall_path)

# -----------------------------
# Figure 11: ONE combined age-stratified beeswarm (3 rows)
# -----------------------------
age_groups = ["3.0–6.9", "7.0–9.9", "10.0–12.9"]
titles = ["Low age (3.0–6.9)", "Mid age (7.0–9.9)", "High age (10.0–12.9)"]

# Use overall mean|SHAP| to select a unified Top-N feature set (fixed order across age groups)
mean_abs_overall = np.abs(shap_values_pos).mean(axis=0)
top_idx = np.argsort(mean_abs_overall)[-TOP_N:][::-1]
feature_names_top = feature_names[top_idx]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.25, right=0.86)

for i, (ax, g, ttl) in enumerate(zip(axes, age_groups, titles)):
    pos = np.where(df["age_group"].values == g)[0]
    if len(pos) == 0:
        ax.axis("off")
        ax.set_title(f"{ttl} (n=0)")
        continue

    X_g = X_trans[pos, :][:, top_idx]
    shap_g = shap_values_pos[pos, :][:, top_idx]

    plt.sca(ax)
    shap.summary_plot(
        shap_g,
        X_g,
        feature_names=feature_names_top,
        max_display=TOP_N,
        show=False,
        color_bar=(i == 2)
    )
    ax.set_title(f"Age {g}", fontsize=12)
    ax.set_xlabel("")

axes[-1].set_xlabel("SHAP value (impact on model output)", fontsize=11)
fig.suptitle("Figure 11. Age-stratified SHAP beeswarm (unified Top features)", fontsize=13, y=0.995)

fig11_path = os.path.join(OUT_DIR, "Figure11_age_stratified_beeswarm_3rows.png")
fig.savefig(fig11_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", fig11_path)

# -----------------------------
# Table 12: Age-stratified Top-5 features (by mean|SHAP|)
# -----------------------------
table12_df = export_topk_by_age_to_excel(
    df=df,
    shap_values_pos=shap_values_pos,
    feature_names=feature_names,
    out_dir=OUT_DIR,
    top_k=TOP_K_TABLE,
    age_groups=tuple(age_groups),
    excel_name="Table12_AgeStratified_Top5_SHAP.xlsx",
    csv_name="Table12_AgeStratified_Top5_SHAP.csv",
)

print("\nDone. Outputs in:", os.path.abspath(OUT_DIR))
