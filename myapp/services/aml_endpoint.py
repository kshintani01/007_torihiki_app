# myapp/services/aml_endpoint.py
import os, json, uuid
import pandas as pd
import numpy as np
import requests
from typing import List, Tuple, Optional

# --- optional: AAD ---
try:
    from azure.identity import DefaultAzureCredential
except Exception:
    DefaultAzureCredential = None

# =============== 環境変数ヘルパ & デバッグ/バッチ設定 ===============
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default

AML_DEBUG = (_env("AML_DEBUG", "0") or "0").strip().lower() not in ("", "0", "false", "off")
OUTPUT_PROBA = (_env("OUTPUT_PROBA", "1") or "1").strip().lower() not in ("0", "false", "no", "off")

AML_BATCH_ROWS = int(_env("AML_BATCH_ROWS", "0"))               # 0 or 負ならバッチ無効
AML_NUM_NULL_FILL = _env("AML_NUM_NULL_FILL", None)             # 例: "0"（数値欠損埋め／任意）
AML_CAT_NULL_FILL = _env("AML_CAT_NULL_FILL", None)             # 例: ""（カテゴリ欠損埋め／任意)

# =============== 有効/無効理由（model_service からも使う） ===============
def aml_enabled_reason() -> str:
    """
    AML を使えない理由を人間可読で返す。空文字なら“有効”。
    必須: AML_SCORING_URI, AML_TEMPLATE_JSON（ファイル存在チェック）
    認証: AML_AUTH_MODE in {"key","aad"}。key の場合 AML_API_KEY 必須
    """
    uri = _env("AML_SCORING_URI")
    tpl = _env("AML_TEMPLATE_JSON")

    if not uri and not tpl:
        return "AML_SCORING_URI と AML_TEMPLATE_JSON が未設定"
    if not uri:
        return "AML_SCORING_URI が未設定"
    if not tpl:
        return "AML_TEMPLATE_JSON が未設定"
    if not os.path.exists(tpl):
        return f"テンプレファイルが存在しません: {tpl}"

    auth = (_env("AML_AUTH_MODE", "key") or "key").strip().lower()
    if auth not in ("key", "aad"):
        return f"AML_AUTH_MODE が不正です: {auth}"
    if auth == "key" and not _env("AML_API_KEY"):
        return "AML_AUTH_MODE=key ですが AML_API_KEY が未設定"

    return ""  # 問題なし

def aml_enabled() -> bool:
    return aml_enabled_reason() == ""

# =============== 前処理ユーティリティ ===============
def _normalize_cols_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [c.strip().strip('"').strip("'") for c in s.split(",") if c.strip()]

def _get_bearer(auth_mode: str, api_key: Optional[str]) -> str:
    if auth_mode == "key":
        if not api_key:
            raise RuntimeError("AML_AUTH_MODE=key ですが AML_API_KEY が未設定です。")
        return api_key
    # aad
    if DefaultAzureCredential is None:
        raise RuntimeError("azure-identity が必要です（pip install azure-identity）。")
    cred = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    return cred.get_token("https://ml.azure.com/.default").token

def _load_template(path: str) -> Tuple[dict, List[str]]:
    if not os.path.exists(path):
        raise RuntimeError(f"AML_TEMPLATE_JSON が見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tpl = json.load(f)
    try:
        cols = tpl["input_data"]["columns"]
        if not isinstance(cols, list) or not cols:
            raise KeyError
        return tpl, cols
    except Exception:
        raise RuntimeError("テンプレJSONに 'input_data.columns' が見つかりません。")

def _shape_for_template(df: pd.DataFrame, expected_cols: List[str], drop_cols_csv: Optional[str]) -> pd.DataFrame:
    drops = _normalize_cols_list(drop_cols_csv)
    if drops:
        exist = [c for c in drops if c in df.columns]
        if exist:
            df = df.drop(columns=exist)

    # 欠け列は None で補完、順序は expected、余計は送らない
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols]

def _apply_null_policy(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """
    任意: 欠損が苦手な scoring スクリプト向けに、簡易に欠損埋めを行う。
    - 数値列判定は dtype ベース（object の列はカテゴリ扱い）
    - 必要な列のみ処理
    """
    if AML_NUM_NULL_FILL is None and AML_CAT_NULL_FILL is None:
        return df
    out = df.copy()
    for c in expected_cols:
        if c not in out.columns:
            continue
        s = out[c]
        if AML_NUM_NULL_FILL is not None and pd.api.types.is_numeric_dtype(s):
            out[c] = s.fillna(pd.to_numeric(AML_NUM_NULL_FILL, errors="coerce"))
        elif AML_CAT_NULL_FILL is not None:
            out[c] = s.fillna(AML_CAT_NULL_FILL)
    return out

def _payload_from_df(df: pd.DataFrame) -> dict:
    arr = df.to_numpy()
    data = []
    for row in arr:
        out = []
        for v in row:
            if pd.isna(v):
                out.append(None); continue
            if isinstance(v, (np.integer,)):
                out.append(int(v)); continue
            if isinstance(v, (np.floating,)):
                out.append(float(v)); continue
            if hasattr(v, "isoformat"):
                try:
                    out.append(v.isoformat()); continue
                except Exception:
                    pass
            out.append(v)
        data.append(out)
    return {"input_data": {"columns": df.columns.tolist(), "data": data}}


# =============== 応答パース & 送信（バッチ対応） ===============
def _parse_response(obj: dict) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[List[str]]]:
    """
    返り値: (classes, proba_matrix, predicted_labels)
    - classes: ["A","B",...]
    - proba_matrix: shape=(n_samples, n_classes)
    - predicted_labels: ["A","B",...]（確率なしのとき）
    """
    if isinstance(obj, list) and obj and all(not isinstance(x, dict) for x in obj):
        return None, None, obj
    
    # 1) {"result": [{"probabilities": {...}, "label": "X"}, ...]}
    if "result" in obj and isinstance(obj["result"], list) and obj["result"]:
        probs_list, labels = [], []
        classes = None
        for r in obj["result"]:
            if isinstance(r, dict) and "probabilities" in r and isinstance(r["probabilities"], dict):
                if classes is None:
                    classes = list(r["probabilities"].keys())
                probs_list.append([float(r["probabilities"].get(c, 0.0)) for c in classes])
            if isinstance(r, dict) and "label" in r:
                labels.append(r["label"])
        if probs_list:
            return classes, np.array(probs_list, dtype=float), (labels if labels else None)

    # 2) {"predictions":[{"labels":[...], "scores":[[...], ...]}]}
    if "predictions" in obj:
        preds = obj["predictions"]
        if isinstance(preds, list) and preds:
            if all(not isinstance(x, dict) for x in preds):
                first = preds[0]
                if isinstance(first, (list, tuple, np.ndarray)):
                    try:
                        labels = [int(np.argmax(np.asarray(r))) if isinstance(r, (list, tuple, np.ndarray)) and len(r) else None for r in preds]
                        return None, None, labels
                    except Exception:
                        return None, None, preds
                return None, None, preds
            p0 = preds[0]
            if isinstance(p0, dict):
                classes = p0.get("labels") or p0.get("classes")
                scores = p0.get("score") or p0.get("probabilities")
                if classes and scores is not None:
                    return classes, np.ndarray(scores, dtype=float), None
    

    # 3) {"labels":[...], "scores":[[...]]}
    if isinstance(obj, dict) and "labels" in obj and "scores" in obj:
        return obj["labels"], np.array(obj["scores"], dtype=float), None

    # 4) 単純なラベル配列 {"predictions": ["A","B",...]} 等
    if isinstance(obj, dict):
        for k in ["predictions", "output", "result"]:
            v = obj.get(k)
            if isinstance(v, list) and v and all(not isinstance(x, dict) for x in v):
                return None, None, v
    return None, None, None

def _score_chunk(scoring_uri: str, shaped: pd.DataFrame, auth_mode: str, api_key: Optional[str], timeout: int) -> dict:
    """
    1チャンク分を送信して JSON を返す。リクエストIDを付けてログと紐付けやすく。
    """
    req_id = str(uuid.uuid4())
    bearer = _get_bearer(auth_mode, api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer}",
        "x-ms-client-request-id": req_id
    }
    payload = _payload_from_df(shaped)

    if AML_DEBUG:
        prev = {"input_data": {"columns": payload["input_data"]["columns"], "data": payload["input_data"]["data"][:2]}}
        print(f"[AML] request-id={req_id} | rows={len(shaped)}")
        print("[AML] Payload preview:", json.dumps(prev, ensure_ascii=False)[:800])

    resp = requests.post(scoring_uri, headers=headers, data=json.dumps(payload), timeout=timeout)
    if AML_DEBUG:
        print(f"[AML] HTTP {resp.status_code} for request-id={req_id}")
    if resp.status_code >= 400:
        raise RuntimeError(f"AML HTTP {resp.status_code} (req {req_id}): {resp.text[:600]}")
    try:
        return resp.json()
    except Exception:
        raise RuntimeError(f"AML 非JSON応答 (req {req_id}): {resp.text[:300]}")

def _to_output_df(src_df: pd.DataFrame, classes, proba, labels) -> pd.DataFrame:
    out = src_df.copy().reset_index(drop=True)
    if labels is not None:
        out["pred_class"] = labels
        return out
    if proba is not None and classes:
        best = proba.argmax(axis=1)
        out["pred_class"] = [classes[i] for i in best]
        # 確率は設定で出す/出さないを制御
        if OUTPUT_PROBA:
            for j, c in enumerate(classes):
                out[f"pred_prob_{c}"] = proba[:, j]
        return out
    raise RuntimeError("AML 応答の解釈に失敗(proba/labels なし)")

def _score_or_split(scoring_uri: str,
                    part_in: pd.DataFrame,
                    part_src: pd.DataFrame,
                    auth_mode: str,
                    api_key: Optional[str],
                    timeout: int) -> pd.DataFrame:
    """
    1チャンク送信→失敗なら半分に割って再試行。最終的に1行でも失敗した行は pred_class=None で返す。
    （呼び出し側でその行だけローカルにフォールバックできるようにする）
    """
    try:
        obj = _score_chunk(scoring_uri, part_in, auth_mode, api_key, timeout)
        classes, proba, labels = _parse_response(obj)
        return _to_output_df(part_src, classes, proba, labels)
    except Exception as e:
        # 行数1なら諦めて「未スコア」として返す（上位でローカルへ）
        if len(part_in) <= 1:
            out = part_src.copy().reset_index(drop=True)
            out["pred_class"] = [None]
            out["aml_error"] = [str(e)[:200]]
            if AML_DEBUG:
                print(f"[AML] row-level failure: {e}")
            return out
        # 半分に割って再試行
        mid = len(part_in) // 2
        left  = _score_or_split(scoring_uri, part_in.iloc[:mid],  part_src.iloc[:mid],  auth_mode, api_key, timeout)
        right = _score_or_split(scoring_uri, part_in.iloc[mid:], part_src.iloc[mid:], auth_mode, api_key, timeout)
        return pd.concat([left, right], axis=0, ignore_index=True)

# =============== 公開エントリポイント ===============
def score_via_aml(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    df を AML に送り、pred_class（＋必要に応じて pred_prob_*）列を付与して返す。
    例外は上位で捕捉し、ローカル推論にフォールバックさせることを想定。
    """
    reason = aml_enabled_reason()
    if reason:
        raise RuntimeError(f"AML disabled: {reason}")

    scoring_uri = _env("AML_SCORING_URI")
    template = _env("AML_TEMPLATE_JSON")
    auth_mode = _env("AML_AUTH_MODE", "key").lower()
    api_key   = _env("AML_API_KEY")
    drop_cols = _env("AML_DROP_COLUMNS", "")
    timeout   = int(_env("AML_TIMEOUT", "60"))

    tpl, expected_cols = _load_template(template)

    if AML_DEBUG:
        print(f"[AML] endpoint={scoring_uri}")
        print(f"[AML] expected_cols={len(expected_cols)}")

    # 列合わせ & 欠損ポリシー
    shaped_all = _shape_for_template(df.copy(), expected_cols, drop_cols)

    # 列差分の可視化（元DF基準）
    if AML_DEBUG:
        missing = [c for c in expected_cols if c not in df.columns]
        extra   = [c for c in df.columns if c not in expected_cols]
        if missing:
            print(f"[AML] MISSING in DF: {missing[:30]}{' ...' if len(missing) > 30 else ''}")
        if extra:
            print(f"[AML] EXTRA   in DF: {extra[:30]}{' ...' if len(extra) > 30 else ''}")

    shaped_all = _apply_null_policy(shaped_all, expected_cols)

    # バッチ無効 or 行数が少なければ一発送信
    batch = AML_BATCH_ROWS
    outs: List[pd.DataFrame] = []
    if not batch or batch <= 0:
        out = _score_or_split(scoring_uri, shaped_all, df, auth_mode, api_key, timeout)
        return out.assign(used_scorer="aml"), {"used": "aml", "endpoint": scoring_uri} 

    for start in range(0, len(shaped_all), batch):
        part_in  = shaped_all.iloc[start:start+batch]
        part_src = df.iloc[start:start+batch]  # 元の見た目で返したいので元DFで out を作る
        part_out = _score_or_split(scoring_uri, part_in, part_src, auth_mode, api_key, timeout)
        outs.append(part_out)

    merged = pd.concat(outs, axis=0, ignore_index=True)
    merged["used_scorer"] = "aml"
    return merged, {"used": "aml", "endpoint": scoring_uri}
