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

def _ensure_data_type_safety(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML モデルへの入力前に、データ型の安全性を確保する。
    - 学習時の数値列/カテゴリ列の定義に従って適切な型に変換
    - すべての NaN 値を適切に処理
    - scikit-learn/XGBoostが期待するデータ形式に統一
    """
    df_safe = df.copy()
    
    try:
        # 学習時の列定義を取得
        num_cols, cat_cols = get_trained_columns()
        
        print(f"デバッグ: 数値列={len(num_cols)}, カテゴリ列={len(cat_cols)}")
        
        # 各列のデータ型を学習時の定義に従って処理
        for col in df_safe.columns:
            try:
                if col in num_cols:
                    # 数値列として処理
                    series = df_safe[col]
                    
                    # まず文字列化してクリーニング
                    if series.dtype == 'object':
                        series = series.astype(str)
                        # 様々な無効値をNaNに統一
                        series = series.replace({
                            'nan': np.nan, 'None': np.nan, 'null': np.nan, 
                            'NULL': np.nan, '': np.nan, ' ': np.nan
                        })
                    
                    # 数値変換を実行
                    df_safe[col] = pd.to_numeric(series, errors='coerce')
                    
                elif col in cat_cols:
                    # カテゴリ列として処理
                    series = df_safe[col]
                    
                    # 文字列として統一
                    series = series.astype(str)
                    
                    # 無効値をNaNに変換（カテゴリ列では'nan'文字列もNaNとして扱う）
                    series = series.replace({
                        'nan': np.nan, 'None': np.nan, 'null': np.nan,
                        'NULL': np.nan
                    })
                    
                    # 空文字列や空白のみの文字列もNaNに
                    series = series.replace(r'^\s*$', np.nan, regex=True)
                    
                    df_safe[col] = series
                    
                else:
                    # 学習時に存在しなかった列（通常はないはず）
                    print(f"警告: 未知の列 {col} を数値として処理")
                    df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce')
                        
            except Exception as col_error:
                print(f"列 {col} の型安全処理に失敗: {col_error}")
                # 失敗した場合は列の用途に応じてデフォルト値を設定
                if col in num_cols:
                    df_safe[col] = np.nan
                else:
                    df_safe[col] = None
                
        # 最終チェック: すべての数値列が本当に数値型になっているか確認
        for col in num_cols:
            if col in df_safe.columns and not pd.api.types.is_numeric_dtype(df_safe[col]):
                print(f"警告: 数値列 {col} が数値型になっていません。強制変換します。")
                df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce')
                
        print(f"デバッグ: 型安全処理完了。最終形状: {df_safe.shape}")
        return df_safe
        
    except Exception as e:
        print(f"データ型安全処理に失敗: {e}")
        print(f"エラー詳細: {type(e).__name__}: {str(e)}")
        # 全体的に失敗した場合は最低限の処理を試行
        try:
            # すべての列を数値変換試行
            df_fallback = df.copy()
            for col in df_fallback.columns:
                df_fallback[col] = pd.to_numeric(df_fallback[col], errors='coerce')
            return df_fallback
        except:
            # 最終手段: 全部NaNのDataFrame
            return pd.DataFrame(np.nan, index=df.index, columns=df.columns)

def _auto_numeric_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    数値らしい列を安全に float に落とす保険。
    - 既に数値 dtype は触らない
    - 単票（1行）の場合: そのセルが数値に変換できれば数値化
    - 複数行の場合: 非欠損の80%以上が数値に変換できれば数値化
    """
    out = df.copy()
    print(f"デバッグ: auto_numeric_cast開始 - 列数: {len(out.columns)}")
    
    for c in out.columns:
        try:
            s = out[c]
            
            # 既に numeric はスキップ
            if pd.api.types.is_numeric_dtype(s):
                continue
            
            print(f"デバッグ: 列 {c} の数値変換を試行 (dtype: {s.dtype})")
            
            # 文字列の空白を NaN に統一（念のため）
            if s.dtype == "object":
                # まず文字列として扱い、'nan'文字列も考慮
                s_str = s.astype(str).str.strip()
                s_str = s_str.replace({
                    "": np.nan, "nan": np.nan, "None": np.nan, 
                    "null": np.nan, "NULL": np.nan, " ": np.nan
                })
                s = s_str
            
            # 数値変換を試行
            coerced = pd.to_numeric(s, errors="coerce")
            nn = s.notna().sum()
            
            if len(out) <= 1:
                # 単票: 1セルでも数値化できれば採用
                if coerced.notna().any():
                    out[c] = coerced
                    print(f"デバッグ: 列 {c} を数値化しました")
                else:
                    # 数値化できない場合は文字列として保持（ただし、NaNは適切に処理）
                    out[c] = s
            else:
                # 複数行: 8割以上が数値化できるなら採用
                if nn > 0 and (coerced.notna().sum() / nn) >= 0.8:
                    out[c] = coerced
                    print(f"デバッグ: 列 {c} を数値化しました")
                else:
                    # 数値化できない場合は文字列として保持
                    out[c] = s
                    
        except Exception as e:
            # 変換に失敗した場合はNaNで埋める
            print(f"列 {c} の数値変換に失敗: {e}")
            out[c] = np.nan
            
    print(f"デバッグ: auto_numeric_cast完了")
    return out

def _normalize_single_row(df: pd.DataFrame) -> pd.DataFrame:
     """
     単票入力の 1 行 DataFrame を正規化：
       - 空文字/空白のみの文字列は NaN
       - Python の None も NaN
       - np.isnan は使わず、pd.isna 系で扱う
       - データ型を安全に処理
    """
     df = df.copy()
     
     # 空文字や空白だけのセルを NaN に
     df = df.replace(r'^\s*$', np.nan, regex=True)
     
     # 'None' 文字列などの紛れがあればここで追加正規化
     df = df.replace({'None': np.nan, 'NULL': np.nan, 'null': np.nan, '': np.nan})
     
     # 数値型への変換を安全に行う
     df = _auto_numeric_cast(df)
     
     # object型カラムで残っているものは文字列として正規化
     for col in df.columns:
         if df[col].dtype == 'object':
             df[col] = df[col].astype(str)
             df[col] = df[col].replace({'nan': np.nan, 'None': np.nan})
     
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
    try:
        names = FEATURE_NAMES_IN
        df2 = df.copy()
        
        print(f"デバッグ: 列整列開始 - 入力列数: {len(df2.columns)}")
        
        # 空文字列をNaNに変換
        df2 = df2.replace(r'^\s*$', np.nan, regex=True)
        df2 = df2.replace({'': np.nan, 'None': np.nan, 'null': np.nan})
        
        if not names:
            print("デバッグ: 学習時列名が不明のため、auto_numeric_castのみ実行")
            return _auto_numeric_cast(df2)
        
        print(f"デバッグ: 学習時列数: {len(names)}")
        
        # 不足している列を一度に追加（断片化回避）
        missing_cols = [col for col in names if col not in df2.columns]
        if missing_cols:
            print(f"デバッグ: 不足列を追加: {len(missing_cols)}個")
            # 一度にすべての不足列を追加
            missing_data = pd.DataFrame({col: [np.nan] * len(df2) for col in missing_cols}, index=df2.index)
            df2 = pd.concat([df2, missing_data], axis=1)
        
        # 指定された列順に並び替え
        result = df2[names]
        print(f"デバッグ: 列整列完了 - 最終列数: {len(result.columns)}")
        
        # 数値型列を安全に処理
        return _auto_numeric_cast(result)
        
    except Exception as e:
        print(f"列の整列処理でエラー: {e}")
        print(f"エラータイプ: {type(e).__name__}")
        # エラーが発生した場合は元のDataFrameを返す
        return _auto_numeric_cast(df.copy())

def _predict_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    try:
        print(f"デバッグ: predict_proba開始 - 入力形状: {X.shape}")
        
        X_aligned = _align_columns(X, estimator)
        print(f"デバッグ: 列整列完了 - 形状: {X_aligned.shape}")
        
        # データ型安全性を確保
        X_safe = _ensure_data_type_safety(X_aligned)
        print(f"デバッグ: 型安全処理完了 - 形状: {X_safe.shape}")
        
        # データ型の詳細確認
        print("デバッグ: 各列のデータ型:")
        for col in X_safe.columns:
            dtype = X_safe[col].dtype
            sample_val = X_safe[col].iloc[0] if len(X_safe) > 0 else "N/A"
            print(f"  {col}: {dtype} (例: {sample_val})")
        
        # NaN値の確認
        nan_cols = X_safe.columns[X_safe.isna().any()].tolist()
        if nan_cols:
            print(f"デバッグ: NaN含有列: {nan_cols}")
        
        # Pipeline 経由でも predict_proba は透過的に呼べる想定
        if hasattr(estimator, "predict_proba"):
            print("デバッグ: predict_proba実行中...")
            proba = estimator.predict_proba(X_safe)
            proba = np.asarray(proba)
            print(f"デバッグ: predict_proba完了 - 確率形状: {proba.shape}")
            if proba.ndim == 1:  # 念のため防御
                proba = np.vstack([1 - proba, proba]).T
            return proba

        # 予備: decision_function から近似
        if hasattr(estimator, "decision_function"):
            print("デバッグ: decision_function使用")
            logits = np.asarray(estimator.decision_function(X_safe))
            if logits.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-logits))
                p0 = 1.0 - p1
                return np.vstack([p0, p1]).T
            z = logits - logits.max(axis=1, keepdims=True)
            ez = np.exp(z)
            return ez / ez.sum(axis=1, keepdims=True)

        # 最終手段: predict を one-hot に
        print("デバッグ: predict使用（最終手段）")
        labels = estimator.predict(X_safe)
        classes = get_classes()  # 下の関数
        idx_map = {c: i for i, c in enumerate(classes)}
        proba = np.zeros((len(labels), len(classes)), dtype=float)
        for r, lab in enumerate(labels):
            j = idx_map.get(lab, 0)
            proba[r, j] = 1.0
        return proba
        
    except Exception as e:
        print(f"予測確率計算中にエラー: {e}")
        print(f"エラータイプ: {type(e).__name__}")
        print(f"エラー詳細: {str(e)}")
        
        # エラー時はダミー確率を返す
        classes = get_classes()
        uniform_proba = np.ones((len(X), len(classes))) / len(classes)
        print(f"デバッグ: ダミー確率を返します - 形状: {uniform_proba.shape}")
        return uniform_proba

def get_classes():
    """
    出力ラベル順を決定。
    - 優先: 保存されたラベル名配列（SAVED_CLASSES）※あなたの日本語ラベル
    - 次点: 推定器の classes_（整数など）
    - どれも無ければ 0..C-1
    """
    try:
        # モデルが利用できない場合のデフォルト
        if MODEL is None:
            return np.array(["不明"])
        
        # 1) 保存済みのラベル名（推奨）
        if SAVED_CLASSES is not None and len(SAVED_CLASSES) > 0:
            # Noneや無効な値を除外
            valid_classes = [cls for cls in SAVED_CLASSES if cls is not None and str(cls).strip() != '']
            if valid_classes:
                return np.array(valid_classes)

        # 2) 推定器 / 最終推定器の classes_
        if hasattr(MODEL, "classes_"):
            classes = MODEL.classes_
            # Noneや無効な値を除外
            valid_classes = [cls for cls in classes if cls is not None and str(cls).strip() != '']
            if valid_classes:
                return np.array(valid_classes)
                
        if hasattr(MODEL, "steps"):
            try:
                last_est = MODEL.steps[-1][1]
                if hasattr(last_est, "classes_"):
                    classes = last_est.classes_
                    # Noneや無効な値を除外
                    valid_classes = [cls for cls in classes if cls is not None and str(cls).strip() != '']
                    if valid_classes:
                        return np.array(valid_classes)
            except Exception:
                pass

        # 3) どうしても無い場合（通常は来ない）
        # ここに来た場合は、predict_proba 実行時の列数から作る
        try:
            dummy = pd.DataFrame([{}])
            proba = _predict_proba(MODEL, dummy)
            C = proba.shape[1]
            return np.array([f"クラス{i}" for i in range(C)])
        except:
            return np.array(["不明"])
            
    except Exception as e:
        print(f"クラス取得でエラー: {e}")
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
    
    try:
        # データを安全に処理
        X = pd.DataFrame([features])
        X = _normalize_single_row(X)
        X = _restrict_to_topk(X, topk)
        
        # 予測実行
        proba = _predict_proba(MODEL, X)[0]
        
        # 予測結果の安全性を確保
        if len(proba) == 0 or not np.isfinite(proba).all():
            raise ValueError("予測確率が無効です")
            
        pred_idx = int(np.argmax(proba))
        
        # クラス名の安全性を確保
        if pred_idx < len(CLASSES):
            pred_class = CLASSES[pred_idx]
            if pred_class is None or str(pred_class).strip() == '':
                pred_class = f"クラス{pred_idx}"
        else:
            pred_class = f"クラス{pred_idx}"
        
        decision = bool(proba[pred_idx] >= float(threshold))
        
        # 確率辞書を安全に作成
        proba_dict = {}
        for i, cls in enumerate(CLASSES):
            if i < len(proba):
                cls_name = str(cls) if cls is not None and str(cls).strip() != '' else f"クラス{i}"
                proba_dict[cls_name] = float(proba[i])
        
        return {
            "pred_class": str(pred_class),
            "proba": proba_dict,
            "threshold": float(threshold),
            "decision": decision,
            "used_topk": int(topk or 0),
        }
    except Exception as e:
        print(f"予測処理でエラーが発生: {e}")
        return {
            "error": f"予測処理でエラーが発生しました: {str(e)}",
            "pred_class": None,
            "proba": {},
            "threshold": float(threshold),
            "decision": False,
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

    try:
        ordered = _load_shap_ordered_features()
        all_feats = get_required_features()
        # 実在しない列名は除外しておく
        selected = [c for c in ordered if c in all_feats][:k]

        # 最終的にモデルへ渡す列全集合を確保（断片化を避けるため一度に作成）
        out = df.copy()
        out = out.replace(r'^\s*$', np.nan, regex=True)
        
        # 不足している列を一度に追加（断片化回避）
        missing_cols = [col for col in all_feats if col not in out.columns]
        if missing_cols:
            # 一度にすべての不足列を追加
            missing_data = pd.DataFrame({col: [np.nan] * len(out) for col in missing_cols}, index=out.index)
            out = pd.concat([out, missing_data], axis=1)
        
        # ★ 非選択の特徴量は NaN に上書きして「効かない」ようにする
        inactive = [c for c in all_feats if c not in selected]
        if inactive:
            out[inactive] = np.nan

        # 列順は学習時に合わせる
        out = out[all_feats]

        # ★ 数値列は float に強制（object落ち対策） - 効率的に処理
        try:
            num_cols, _ = get_trained_columns()
            existing_num = [c for c in num_cols if c in out.columns]
            if existing_num:
                # 一度にすべての数値列を変換
                for col in existing_num:
                    try:
                        out.loc[:, col] = pd.to_numeric(out[col], errors='coerce')
                    except Exception as e:
                        print(f"列 {col} の数値変換に失敗: {e}")
                        # 変換に失敗した場合はNaNで埋める
                        out.loc[:, col] = np.nan
            else:
                out = _auto_numeric_cast(out)
        except Exception as e:
            print(f"数値列変換処理に失敗: {e}")
            out = _auto_numeric_cast(out)

        # DataFrame を最終的にデフラグメント
        return out.copy()
        
    except Exception as e:
        print(f"topk制限処理でエラー: {e}")
        # エラー時は元のDataFrameを返す
        return _auto_numeric_cast(df.copy())

