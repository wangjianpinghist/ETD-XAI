# -*- coding: utf-8 -*-
"""
Stage-2 Severity Grading (mild/moderate/severe) with XGBoost + SHAP
Outputs:
- Figure 7A: SHAP global feature importance (Top 20) using mean |SHAP|
- Figure 7B: SHAP summary beeswarm (Top features)

Assumption:
- CSV is in the SAME folder as this script.
- File name: simulated_178_with_severity.csv
- Columns:
    label: 1=TD, 0=control
    severity: 0/1/2 for TD only (control rows may be -1)
    severity_text: optional
    sample_id: optional
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier


def read_data(csv_name: str) -> pd.DataFrame:
    if not os.path.exists(csv_name):
        raise FileNotFoundError(
            f"找不到文件：{csv_name}\n"
            f"请确认 CSV 与本脚本在同一目录，或把 csv_name 改为正确路径。"
        )
    return pd.read_csv(csv_name)


def prepare_stage2(df: pd.DataFrame):
    for col in ["label", "severity"]:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必需列：{col}")

    # 第二阶段只用TD样本（label=1）
    df_td = df[df["label"] == 1].copy()
    if df_td.empty:
        raise ValueError("未找到 label=1 的 TD 样本，无法进行第二阶段分级。")

    y = df_td["severity"].astype(int)
    bad = set(y.unique()) - {0, 1, 2}
    if bad:
        raise ValueError(f"TD样本 severity 应为 0/1/2，但发现异常取值：{sorted(list(bad))}")

    drop_cols = [c for c in ["sample_id", "label", "severity", "severity_text"] if c in df_td.columns]
    X = df_td.drop(columns=drop_cols).copy()

    # ✅ 关键修复：把所有“非数值列”统一编码成数值（包括 sex、batch）
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"仍有非数值特征列：{non_numeric}")

    return X, y


def train_xgb_multiclass(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X, y)
    return model


def compute_shap(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=2)  # (n, p, k)
    else:
        shap_array = np.array(shap_values)
        if shap_array.ndim != 3:
            raise ValueError(f"SHAP 输出维度异常：{shap_array.shape}，期望 (n, p, k)")
    return shap_array


def plot_fig7a_global_importance(shap_array: np.ndarray, X: pd.DataFrame, top_n: int = 20, save_path: str = None):
    mean_abs = np.mean(np.abs(shap_array), axis=(0, 2))  # (p,)
    imp = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)

    imp_top = imp.iloc[:top_n][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(imp_top.index, imp_top.values)
    plt.xlabel("mean |SHAP|")
    plt.ylabel("Feature")
    plt.title("Figure 7A. SHAP Global Feature Importance (Top 20)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_fig7b_beeswarm(shap_array: np.ndarray, X: pd.DataFrame, max_display: int = 20, save_path: str = None):
    shap_agg = np.mean(shap_array, axis=2)  # (n, p)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_agg,
        X,
        plot_type="dot",
        max_display=max_display,
        show=False
    )
    plt.title("Figure 7B. SHAP Summary (Beeswarm) - Top Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    csv_name = "simulated_178_with_severity.csv"

    save_fig7a = "Figure7A_SHAP_GlobalTop20.png"
    save_fig7b = "Figure7B_SHAP_Beeswarm.png"

    print(f"[1/5] 读取数据：{csv_name}")
    df = read_data(csv_name)

    print("[2/5] 准备第二阶段数据（仅TD，severity=0/1/2）")
    X, y = prepare_stage2(df)
    print(f"    TD样本数：{len(y)}；特征数：{X.shape[1]}")

    print("[3/5] 训练XGBoost多分类模型（轻/中/重）")
    model = train_xgb_multiclass(X, y)

    print("[4/5] 计算SHAP值")
    shap_array = compute_shap(model, X)
    print(f"    SHAP shape = {shap_array.shape}  (n_samples, n_features, n_classes)")

    print("[5/5] 绘图：Figure 7A / Figure 7B")
    plot_fig7a_global_importance(shap_array, X, top_n=20, save_path=save_fig7a)
    plot_fig7b_beeswarm(shap_array, X, max_display=20, save_path=save_fig7b)

    print("\n完成！已输出图：")
    print(f" - {save_fig7a}")
    print(f" - {save_fig7b}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n程序运行出错：")
        print(str(e))
        sys.exit(1)
