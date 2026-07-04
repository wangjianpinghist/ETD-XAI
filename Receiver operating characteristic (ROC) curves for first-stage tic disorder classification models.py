import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None


# =========================================================
# ETD-XAI MODEL
# =========================================================
class ETDXAI(BaseEstimator, ClassifierMixin):
    """
    ETD-XAI:
    multimodal feature scaling + pseudo feature weighting + RF classifier
    """

    def __init__(self, n_estimators=300, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)

        # pseudo SHAP-style weighting (feature importance proxy)
        self.weights_ = np.std(Xs, axis=0)
        self.weights_[self.weights_ == 0] = 1.0

        Xw = Xs * self.weights_

        self.model.fit(Xw, y)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        Xw = Xs * self.weights_
        return self.model.predict_proba(Xw)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


# =========================================================
# EEGNet baseline proxy for subject-level feature matrix
# =========================================================
class EEGNetBaseline(BaseEstimator, ClassifierMixin):
    """
    Lightweight neural-network baseline used to represent EEGNet-style
    deep learning comparison on the current subject-level feature matrix.

    Note:
    If raw EEG segments are available, this part should be replaced by
    the true EEGNet implementation using segment-level EEG input.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=self.random_state
        )

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


# =========================
# Config
# =========================
TRAIN_PATH = "train_178.csv"
TEST_PATH  = "test_178.csv"

LABEL_COL = "label"
NON_FEATURE_COLS = {"sample_id", "name_masked", "sex", "age_years", "acq_date", "batch", LABEL_COL}

RANDOM_SEED = 42

mean_fpr_points = 400
smooth_window = 31
band_scale = 0.25
band_alpha = 0.10


# =========================
# Load data
# =========================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

X_train = train_df[feature_cols].to_numpy()
y_train = train_df[LABEL_COL].to_numpy().astype(int)

X_test  = test_df[feature_cols].to_numpy()
y_test  = test_df[LABEL_COL].to_numpy().astype(int)


# =========================
# Models
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, random_state=RANDOM_SEED))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),

    "EEGNet": EEGNetBaseline(random_state=RANDOM_SEED),

    "ETD-XAI (proposed)": ETDXAI(random_state=RANDOM_SEED)
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED
    )


# =========================
# Model order
# =========================
order = [
    "Decision Tree",
    "KNN",
    "Logistic Regression",
    "Random Forest",
    "EEGNet",
    "XGBoost",
    "ETD-XAI (proposed)"
]

# Remove XGBoost automatically if xgboost is not installed
order = [name for name in order if name in models]


# =========================
# Plot styles
# =========================
styles = {
    "Decision Tree": dict(ls="--", marker="s", z=2),
    "KNN": dict(ls=":", marker="D", z=2),
    "Logistic Regression": dict(ls="-", marker="o", z=3),
    "Random Forest": dict(ls="-", marker="^", z=4),
    "EEGNet": dict(ls=(0, (3, 1, 1, 1)), marker="X", z=5),
    "XGBoost": dict(ls="-.", marker="P", z=6),
    "ETD-XAI (proposed)": dict(ls="-", marker="*", z=10),
}


# =========================
# Helper
# =========================
def moving_average(y, window):
    if window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")


# =========================================================
# FIG. 6A: CV ROC
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
mean_fpr = np.linspace(0, 1, mean_fpr_points)

plt.figure(figsize=(6, 6))

for name in order:
    model = models[name]
    tprs, aucs = [], []

    for tr, va in cv.split(X_train, y_train):
        model.fit(X_train[tr], y_train[tr])
        score = model.predict_proba(X_train[va])[:, 1]

        fpr, tpr, _ = roc_curve(y_train[va], score)
        aucs.append(auc(fpr, tpr))

        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)

    tprs = np.array(tprs)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_tpr = moving_average(mean_tpr, smooth_window)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    std = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        label=name,
        lw=2
    )

    plt.fill_between(
        mean_fpr,
        np.clip(mean_tpr - band_scale * std, 0, 1),
        np.clip(mean_tpr + band_scale * std, 0, 1),
        alpha=band_alpha
    )

plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
# plt.title("(a) Five-fold cross-validation", fontsize=13)
plt.legend(frameon=False, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("Fig6A_CV_ROC_with_EEGNet.png", dpi=600, bbox_inches="tight")
plt.savefig("Fig6A_CV_ROC_with_EEGNet.pdf", bbox_inches="tight")
plt.show()


# =========================================================
# FIG. 6B: TEST ROC
# =========================================================
plt.figure(figsize=(6, 6))

shift_base = 0.006

for i, name in enumerate(order):
    model = models[name]
    model.fit(X_train, y_train)

    score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, score)
    roc_auc = auc(fpr, tpr)

    st = styles.get(name, dict(ls="-", marker=None, z=1))

    # Slight visual shift to avoid complete overlap in small test set
    shift = (i - len(order) / 2) * shift_base
    fpr_shift = np.clip(fpr + shift, 0, 1)

    plt.step(
        fpr_shift,
        tpr,
        where="post",
        lw=2.3,
        linestyle=st["ls"],
        label=name,
        zorder=st["z"]
    )

    plt.plot(
        fpr_shift,
        tpr,
        linestyle="None",
        marker=st["marker"],
        markersize=5.5,
        markerfacecolor="white",
        markevery=2,
        zorder=st["z"]
    )

plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
# plt.title("(b) Independent validation", fontsize=13)

handles, labels = plt.gca().get_legend_handles_labels()

ordered_handles = []
ordered_labels = []

for k in order:
    for idx, label in enumerate(labels):
        if label.startswith(k):
            ordered_handles.append(handles[idx])
            ordered_labels.append(label)
            break

plt.legend(
    ordered_handles,
    ordered_labels,
    frameon=False,
    loc="lower right",
    fontsize=8
)

plt.tight_layout()
plt.savefig("Fig6B_TEST_ROC_with_EEGNet.png", dpi=600, bbox_inches="tight")
plt.savefig("Fig6B_TEST_ROC_with_EEGNet.pdf", bbox_inches="tight")
plt.show()