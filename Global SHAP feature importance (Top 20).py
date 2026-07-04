# -*- coding: utf-8 -*-
"""
Fix for: ValueError could not convert string to float: 'M'
- Treat all non-numeric columns as categorical (object/category/bool)
- Coerce numeric columns to float; impute missing by median
- ONE model on full filtered cohort; SHAP overall + age strata

pip install pandas numpy scikit-learn xgboost shap matplotlib pillow
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

from PIL import Image


CSV_PATH = "master_178.csv"
OUT_DIR = "shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

AGE_MIN, AGE_MAX = 3.0, 12.9
TOP_N = 20
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

# -----------------------------
# Build X/y
# -----------------------------
target_col = "label"
DROP_COLS = ["sample_id", "acq_date", "name_masked", "age_group"]
drop_exist = [c for c in DROP_COLS if c in df.columns]

X = df.drop(columns=drop_exist + [target_col]).copy()
y = df[target_col].astype(int).values

# ✅ Robust column typing:
# Treat "non-numeric" as categorical, regardless of dtype weirdness.
numeric_cols = []
categorical_cols = []

for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

print("Detected numeric cols:", len(numeric_cols))
print("Detected categorical cols:", categorical_cols)

# Coerce numeric cols to float (in case some are stored as strings)
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
    tree_method="hist",  # stable on Windows
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
# Figure 8A–C (age strata)
# -----------------------------
age_groups = ["3.0–6.9", "7.0–9.9", "10.0–12.9"]
panel_paths = []

for i, g in enumerate(age_groups, start=1):
    pos = np.where(df["age_group"].values == g)[0]
    X_g = X_trans[pos, :]
    shap_g = shap_values_pos[pos, :]

    out_path = os.path.join(OUT_DIR, f"Figure8_{chr(64+i)}_beeswarm_{g}.png")
    save_beeswarm(
        shap_g,
        X_g,
        feature_names,
        out_path,
        title=f"Figure 8{chr(64+i)}. SHAP beeswarm (Age {g})",
        max_display=TOP_N
    )
    panel_paths.append(out_path)
    print("Saved:", out_path, "| n =", X_g.shape[0])

# Combine A/B/C
combined_path = os.path.join(OUT_DIR, "Figure8_ABC_combined.png")
imgs = [Image.open(p).convert("RGB") for p in panel_paths]
min_h = min(im.size[1] for im in imgs)

resized = []
for im in imgs:
    w, h = im.size
    new_w = int(w * (min_h / h))
    resized.append(im.resize((new_w, min_h)))

total_w = sum(im.size[0] for im in resized)
canvas = Image.new("RGB", (total_w, min_h), (255, 255, 255))

x = 0
for im in resized:
    canvas.paste(im, (x, 0))
    x += im.size[0]

canvas.save(combined_path, quality=95)
print("Saved:", combined_path)
print("\nDone. Outputs in:", os.path.abspath(OUT_DIR))
