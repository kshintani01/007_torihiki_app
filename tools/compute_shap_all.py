#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済みジョブリブ（{"pipeline": pipe, "classes": ...}）から SHAP を計算し、
(1) 変換後特徴の重要度、(2) 元の正規特徴（num/cat）に集約した重要度を算出。
全特徴量を降順に並べて JSON に保存します（TopK 制限なし）。

実行方法: そのまま `python tools/compute_shap_all.py`
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

# =========================
# 設定（ここだけ編集すればOK）
# =========================
CONFIG = dict(
    MODEL_PATH="models/xgb_std_pipeline.joblib",   # あなたの保存名に合わせています
    DATA_PATH="/Users/kshintani/Documents/ソフトバンク/03_CI本部/KE/007_torihiki/007_torihiki_app/取引審査データ_250829_train_data.csv",
    TARGET_COL="審査結果",
    DROP_COLS=["使用許可", "許可番号", "BIS申請", "審査結果(1)", "審査結果(2)"],
    MAX_SAMPLE=2000,                                # 計算負荷対策（データが小さければ大きくして可）
    OUTPUT_DIR="config",
    RANDOM_STATE=42,
    MODEL_OUTPUT="raw",                             # "raw" or "probability"（重い時は raw 推奨）
)
# =========================

BASE_DIR = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "tools") else Path.cwd()
MODEL_PATH = (BASE_DIR / CONFIG["MODEL_PATH"]).resolve()
DATA_PATH  = Path(CONFIG["DATA_PATH"]).resolve()
OUTDIR     = (BASE_DIR / CONFIG["OUTPUT_DIR"]).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_pipeline(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj  # Pipeline そのものが保存されている場合

def prepare_dataframe(csv_path: Path, target_col: str, drop_cols):
    df = pd.read_csv(csv_path, low_memory=False)
    # 目的変数/不要列は落とす
    cols_to_drop = [c for c in ([target_col] + drop_cols) if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore").copy()

    # 文字列を整形（トリム、空→NaN）
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in obj_cols:
        s = X[c].astype(str).str.strip()
        X[c] = s.replace({"": np.nan})

    # 「数値っぽい」文字列を数値へ（7割以上数値化できれば採用）
    def try_to_numeric(s: pd.Series) -> pd.Series:
        s2 = (s.astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("¥", "", regex=False))
        num = pd.to_numeric(s2, errors="coerce")
        return num if num.notna().mean() >= 0.7 else s

    for c in obj_cols:
        X[c] = try_to_numeric(X[c])

    return X

def get_feature_names_after_prep(prep, X_like: pd.DataFrame):
    # sklearn 1.1+ を想定
    try:
        names = prep.get_feature_names_out()
        return list(map(str, names))
    except Exception as e:
        raise RuntimeError("prep.get_feature_names_out() が利用できません。scikit-learn>=1.1 を推奨。") from e

def get_cat_cols_from_prep(prep):
    # fit 後は transformers_、未fit なら transformers
    trans_list = getattr(prep, "transformers_", None) or getattr(prep, "transformers", [])
    cat_cols = []
    for name, trans, cols in trans_list:
        if name == "cat":
            cat_cols = list(cols)
    return cat_cols

def canonical_name_from_transformed(name: str, cat_cols: list[str]) -> str:
    # ColumnTransformer 命名規則を元に元列へ集約
    # num: "num__<col>"
    # cat: "cat__<col>_<category>"（<col> に '_' を含む場合がある → 最長一致）
    if name.startswith("num__"):
        return name.split("num__", 1)[1]
    if name.startswith("cat__"):
        tail = name.split("cat__", 1)[1]
        # 最長一致で <col> を見つける
        matches = [c for c in cat_cols if tail == c or tail.startswith(c + "_")]
        if matches:
            return sorted(matches, key=len, reverse=True)[0]
        return tail.split("_", 1)[0]
    return name

def mean_abs_shap_importance(shap_values) -> np.ndarray:
    """
    変換後特徴ごとの |SHAP| 平均を返す（多クラスはクラス軸も平均）。
    """
    vals = shap_values
    if isinstance(vals, list):
        # list of (n_samples, n_features) per class
        arr = np.stack([np.abs(v).mean(axis=0) for v in vals], axis=0)  # (n_classes, n_features)
        return arr.mean(axis=0)
    if hasattr(vals, "values"):
        v = vals.values
    else:
        v = np.asarray(vals)
    if v.ndim == 3:
        return np.abs(v).mean(axis=(0, 2))
    elif v.ndim == 2:
        return np.abs(v).mean(axis=0)
    else:
        raise ValueError(f"Unexpected SHAP shape: {v.shape}")

def main():
    print(f"[LOAD] MODEL: {MODEL_PATH}")
    pipe = load_pipeline(MODEL_PATH)
    try:
        prep = pipe.named_steps["prep"]
        clf  = pipe.named_steps["clf"]
    except Exception as e:
        raise RuntimeError("パイプラインに 'prep'（ColumnTransformer）と 'clf'（推定器）が必要です。") from e

    print(f"[LOAD] DATA : {DATA_PATH}")
    X = prepare_dataframe(DATA_PATH, CONFIG["TARGET_COL"], CONFIG["DROP_COLS"])
    if len(X) == 0:
        raise ValueError("データが空です。DATA_PATH を見直してください。")

    # サンプリング（重い計算を避ける）
    n = min(CONFIG["MAX_SAMPLE"], len(X))
    Xs = X.sample(n=n, random_state=CONFIG["RANDOM_STATE"])
    print(f"[INFO] sample={n}/{len(X)} rows for SHAP")

    # 変換＆列名
    print("[TRANSFORM] preprocessing...")
    X_trans = prep.transform(Xs)
    if hasattr(X_trans, "toarray"):  # 念のため
        X_trans = X_trans.toarray()
    feat_names_trans = get_feature_names_after_prep(prep, Xs)
    if X_trans.shape[1] != len(feat_names_trans):
        raise RuntimeError(f"変換後の列数不一致: X_trans={X_trans.shape[1]} names={len(feat_names_trans)}")

    # SHAP 計算（XGBoost は TreeExplainer が高速・安定）
    print("[SHAP] computing...")
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent",
                                   model_output=CONFIG["MODEL_OUTPUT"])
    shap_values = explainer.shap_values(X_trans)
    imp_trans = mean_abs_shap_importance(shap_values)  # 変換後特徴の重要度（|SHAP|平均）

    # 変換後 → 元列へ集約
    print("[AGG] aggregate to canonical features...")
    cat_cols = get_cat_cols_from_prep(prep)
    group_imp = {}
    for name, imp in zip(feat_names_trans, imp_trans):
        key = canonical_name_from_transformed(name, cat_cols)
        group_imp[key] = group_imp.get(key, 0.0) + float(imp)

    # 降順ソート（全特徴量）
    ordered = sorted(group_imp.items(), key=lambda kv: kv[1], reverse=True)
    ordered_names = [k for k, _ in ordered]

    # 保存
    out1 = OUTDIR / "shap_feature_importance.json"        # 元列ごとの重要度（降順）
    out2 = OUTDIR / "shap_ordered_features.json"          # 重要度降順の列名だけ
    out3 = OUTDIR / "shap_transformed_importance.json"    # 変換後（OneHot後）列の重要度（デバッグ）

    out1.write_text(json.dumps({"importance": ordered}, ensure_ascii=False, indent=2), encoding="utf-8")
    out2.write_text(json.dumps({"ordered": ordered_names}, ensure_ascii=False, indent=2), encoding="utf-8")
    out3.write_text(json.dumps(
        {"importance": list(zip(feat_names_trans, list(map(float, imp_trans))))},
        ensure_ascii=False, indent=2
    ), encoding="utf-8")

    print(f"[OK] saved: {out1}")
    print(f"[OK] saved: {out2}")
    print(f"[OK] saved: {out3}")

if __name__ == "__main__":
    main()
