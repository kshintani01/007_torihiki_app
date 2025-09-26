# predictor/model_service.py
import os, io, json, joblib
import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from .services.blob_store import download_joblib_bytes, download_text, models_path, config_path
from typing import Optional 

def _resolve_model_blob_path() -> str:
    return os.getenv("MODEL_BLOB_PATH") or models_path("xgb_std_pipeline.joblib")

def _auto_numeric_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    数値らしい列を安全に float に落とす保険。
    - 既に数値 dtype は触らない
    - 単票（1行）の場合: そのセルが数値に変換できれば数値化
    - 複数行の場合: 非欠損の80%以上が数値に変換できれば数値化
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        # 既に numeric はスキップ
        if pd.api.types.is_numeric_dtype(s):
            continue
        # 文字列の空白を NaN に統一（念のため）
        if s.dtype == "object":
            s = s.astype(str).str.strip().replace({"": np.nan})
        coerced = pd.to_numeric(s, errors="coerce")
        nn = s.notna().sum()
        if len(out) <= 1:
            # 単票: 1セルでも数値化できれば採用
            if coerced.notna().any():
                out[c] = coerced
        else:
            # 複数行: 8割以上が数値化できるなら採用
            if nn > 0 and (coerced.notna().sum() / nn) >= 0.8:
                out[c] = coerced
    return out

def _normalize_single_row(df: pd.DataFrame) -> pd.DataFrame:
     """
     単票入力の 1 行 DataFrame を正規化：
       - 空文字/空白のみの文字列は NaN
       - Python の None も NaN
       - np.isnan は使わず、pd.isna 系で扱う
    """
     # 空文字や空白だけのセルを NaN に
     df = df.replace(r'^\s*$', np.nan, regex=True)
     # 'None' 文字列などの紛れがあればここで追加正規化しても良い:
     # df = df.replace({'None': np.nan, 'NULL': np.nan})
     df = _auto_numeric_cast(df)
     return df

def _load_model_and_classes():
    try:
        blob_path = _resolve_model_blob_path()
        bio = download_joblib_bytes(blob_path)
        obj = joblib.load(bio)

        # A) dict 形式（あなたの train スクリプトの保存形式）
        if isinstance(obj, dict):
            if "pipeline" not in obj:
                raise ValueError("model joblib は dict ですが 'pipeline' キーがありません。")
            pipeline = obj["pipeline"]
            classes = obj.get("classes", None)  # 学習時ラベル名（推奨）
            return pipeline, classes

        # B) それ以外は推定器とみなす
        estimator = obj
        classes = None
        # 推定器 or 最終推定器から classes_ を拾えるなら拾う
        if hasattr(estimator, "classes_"):
            classes = list(estimator.classes_)
        elif hasattr(estimator, "steps"):  # Pipeline
            try:
                last_est = estimator.steps[-1][1]
                if hasattr(last_est, "classes_"):
                    classes = list(last_est.classes_)
            except Exception:
                pass
        return estimator, classes
    except Exception as e:
        print(f"警告: モデルファイルの読み込みに失敗しました(Blob)")
        print(f"エラー: {e}")
        print("アプリケーションは起動しますが、予測機能は無効です。")
        return None, None

MODEL, SAVED_CLASSES = _load_model_and_classes()

def _get_feature_names_in(estimator):
    """
    学習時の入力列順が取れるなら取得（なければ None）
    Pipeline/推定器の feature_names_in_ を優先して拾う
    """
    names = getattr(estimator, "feature_names_in_", None)
    if names is None and hasattr(estimator, "steps"):
        try:
            last_est = estimator.steps[-1][1]
            names = getattr(last_est, "feature_names_in_", None)
        except Exception:
            pass
    return list(names) if names is not None else None

FEATURE_NAMES_IN = _get_feature_names_in(MODEL)

def _align_columns(df: pd.DataFrame, estimator):
    """
    学習時列順が分かる場合はそれに合わせる。不足列は NaN で埋める。余分列は捨てる。
    """
    names = FEATURE_NAMES_IN
    df2 = df.copy()
    df2 = df2.replace(r'^\s*$', np.nan, regex=True)
    if not names:
        return _auto_numeric_cast(df2)
    for col in names:
        if col not in df2.columns:
            df2[col] = np.nan
    return df2[names]

def _predict_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    X_aligned = _align_columns(X, estimator)
    # Pipeline 経由でも predict_proba は透過的に呼べる想定
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_aligned)
        proba = np.asarray(proba)
        if proba.ndim == 1:  # 念のため防御
            proba = np.vstack([1 - proba, proba]).T
        return proba

    # 予備: decision_function から近似
    if hasattr(estimator, "decision_function"):
        logits = np.asarray(estimator.decision_function(X_aligned))
        if logits.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-logits))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)

    # 最終手段: predict を one-hot に
    labels = estimator.predict(X_aligned)
    classes = get_classes()  # 下の関数
    idx_map = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(labels), len(classes)), dtype=float)
    for r, lab in enumerate(labels):
        j = idx_map.get(lab, 0)
        proba[r, j] = 1.0
    return proba

def get_classes():
    """
    出力ラベル順を決定。
    - 優先: 保存されたラベル名配列（SAVED_CLASSES）※あなたの日本語ラベル
    - 次点: 推定器の classes_（整数など）
    - どれも無ければ 0..C-1
    """
    # モデルが利用できない場合のデフォルト
    if MODEL is None:
        return np.array(["不明"])
    
    # 1) 保存済みのラベル名（推奨）
    if SAVED_CLASSES is not None and len(SAVED_CLASSES) > 0:
        return np.array(SAVED_CLASSES)

    # 2) 推定器 / 最終推定器の classes_
    if hasattr(MODEL, "classes_"):
        return np.array(MODEL.classes_)
    if hasattr(MODEL, "steps"):
        try:
            last_est = MODEL.steps[-1][1]
            if hasattr(last_est, "classes_"):
                return np.array(last_est.classes_)
        except Exception:
            pass

    # 3) どうしても無い場合（通常は来ない）
    # ここに来た場合は、predict_proba 実行時の列数から作る
    try:
        dummy = pd.DataFrame([{}])
        proba = _predict_proba(MODEL, dummy)
        C = proba.shape[1]
        return np.arange(C)
    except:
        return np.array(["不明"])

CLASSES = get_classes()

def predict_one(features: dict, threshold: float = 0.5, topk: Optional[int] = None):
    if MODEL is None:
        return {
            "error": "モデルが利用できません。モデルファイルの読み込みに失敗している可能性があります。",
            "pred_class": None,
            "proba": {},
            "threshold": float(threshold),
            "decision": False,
            "used_topk": int(topk or 0),
        }
    
    X = pd.DataFrame([features])
    X = _normalize_single_row(X)
    X = _restrict_to_topk(X, topk)
    proba = _predict_proba(MODEL, X)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = CLASSES[pred_idx]
    decision = bool(proba[pred_idx] >= float(threshold))
    return {
        "pred_class": str(pred_class),
        "proba": {str(CLASSES[i]): float(proba[i]) for i in range(len(CLASSES))},
        "threshold": float(threshold),
        "decision": decision,
        "used_topk": int(topk or 0),
    }

def predict_df(df: pd.DataFrame, topk: Optional[int] = None):
    if MODEL is None:
        out = df.copy()
        out["pred_class"] = "モデル利用不可"
        out["error"] = "モデルが利用できません"
        out["used_topk"] = int(topk or 0)
        return out
    
    X = df.replace(r'^\s*$', np.nan, regex=True)
    X = _auto_numeric_cast(X)
    X = _restrict_to_topk(X, topk)
    proba = _predict_proba(MODEL, X)
    pred_idx = proba.argmax(axis=1)
    pred_class = [str(CLASSES[i]) for i in pred_idx]
    out = df.copy()
    out["pred_class"] = pred_class
    for i, c in enumerate(CLASSES):
        out[f"pred_prob_{c}"] = proba[:, i]
    out["used_topk"] = int(topk or 0)
    return out


def get_trained_columns():
    """
    学習済み Pipeline 内の ColumnTransformer から
    数値カラム/カテゴリカラムのリストを取り出す。
    """
    num_cols, cat_cols = [], []
    # Pipeline の 'prep' ステップを想定
    ct = None
    if hasattr(MODEL, "named_steps") and "prep" in MODEL.named_steps:
        ct = MODEL.named_steps["prep"]
    if ct is None:
        return num_cols, cat_cols

    # fit 後は transformers_、未fitなら transformers
    trans_list = getattr(ct, "transformers_", None) or getattr(ct, "transformers", [])
    for name, trans, cols in trans_list:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    return num_cols, cat_cols

def get_required_features():
    """学習時の入力列（順序つき）を返す。"""
    if FEATURE_NAMES_IN:
        return list(FEATURE_NAMES_IN)
    num_cols, cat_cols = get_trained_columns()
    # 重複排除して順序維持
    return list(dict.fromkeys([*num_cols, *cat_cols]))

def _load_shap_ordered_features():
    """
    Blob 上の config/shap_ordered_features.json を読み、重要度降順の列名リストを返す。
    無ければ学習時の全特徴量順序にフォールバック。
    期待フォーマット:
      {"ordered": ["featA","featB",...]}
    """
    blob_path = os.getenv("SHAP_ORDER_BLOB_PATH") or config_path("shap_ordered_features.json")
    try:
        text = download_text(blob_path)
        data = json.loads(text)
        ordered = data.get("ordered") or data.get("topk")  # 互換
        if isinstance(ordered, list) and len(ordered) > 0:
            return ordered
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return get_required_features()

def _restrict_to_topk(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
    """
    上位Kの特徴量以外は NaN にして無効化する。
    K が None/0/負値 の場合は全列をそのまま使用。
    ※モデルは ColumnTransformer で全列を期待するので列自体は残す。
    """
    if not k or k <= 0:
        # 何もしない
        return df

    ordered = _load_shap_ordered_features()
    all_feats = get_required_features()
    # 実在しない列名は除外しておく
    selected = [c for c in ordered if c in all_feats][:k]

    # 最終的にモデルへ渡す列全集合を確保
    out = df.copy()
    out = out.replace(r'^\s*$', np.nan, regex=True)
    for col in all_feats:
        if col not in out.columns:
            out[col] = np.nan
    
    # ★ 非選択の特徴量は NaN に上書きして「効かない」ようにする
    inactive = [c for c in all_feats if c not in selected]
    if inactive:
        out[inactive] = np.nan

    # 列順は学習時に合わせる
    out = out[all_feats]

    # ★ 数値列は float に強制（object落ち対策）
    num_cols, _ = get_trained_columns()
    existing_num = [c for c in num_cols if c in out.columns]
    if existing_num:
        out[existing_num] = out[existing_num].apply(
            lambda s: pd.to_numeric(s, errors='coerce')
        )
    else:
        out = _auto_numeric_cast(out)

    return out

