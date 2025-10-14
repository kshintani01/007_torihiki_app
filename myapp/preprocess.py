# predictor/preprocess.py
import numpy as np
import pandas as pd

def _get_trained_columns_lazy():
    try:
        from .model_service import get_trained_columns as _gtc
        num_cols, cat_cols = _gtc()
        return list(num_cols), list(cat_cols)
    except Exception:
        return [], []

def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """全 object 列: 前後空白を除去、空文字は NaN に。"""
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        df[c] = s.replace({"": np.nan})
    return df

def _to_numeric_like_series(s: pd.Series) -> pd.Series:
    """
    学習時と同じ発想で、カンマ/パーセントを除去して数値化（失敗は NaN）。
    ※学習スクリプトでは 0.7 しきい値判定がありましたが、
      ここでは「数値カラムに対しては必ず数値化」を行います。
    """
    s2 = s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    # 失敗は NaN（pandas の nullable float に）
    return pd.to_numeric(s2, errors="coerce")

def _apply_numeric_cast_for_trained_num_cols(df: pd.DataFrame) -> pd.DataFrame:
    """学習時に『数値』として扱われた列にだけ、数値化処理を適用。"""
    df = df.copy()
    num_cols, _ = _get_trained_columns_lazy()
    for c in num_cols:
        if c in df.columns:
            df[c] = _to_numeric_like_series(df[c])
    return df

def preprocess_record(d: dict) -> pd.DataFrame:
    """
    単票入力 → DataFrame 1行
    学習時の『数値カラム』には数値化を適用。
    """
    df = pd.DataFrame([d])
    df = _strip_strings(df)
    df = _apply_numeric_cast_for_trained_num_cols(df)
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV入力 → DataFrame
    学習時の『数値カラム』には数値化を適用。
    """
    df = _strip_strings(df)
    df = _apply_numeric_cast_for_trained_num_cols(df)
    return df
