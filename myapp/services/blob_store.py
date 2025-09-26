import os
import io
from typing import Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# 認証は以下のいずれかで
# - AZURE_STORAGE_CONNECTION_STRING
# - BLOB_ACCOUNT_URL + BLOB_SAS_TOKEN
# - BLOB_ACCOUNT_URL + (Managed Identity)  ※この例では未実装。必要なら azure-identity を使って拡張

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
ACCOUNT_URL = os.getenv("BLOB_ACCOUNT_URL")           # 例: https://<acct>.blob.core.windows.net
SAS_TOKEN = os.getenv("BLOB_SAS_TOKEN")               # 例: ?sv=...

CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "appdata")
MODELS_PREFIX = os.getenv("BLOB_MODELS_PREFIX", "models")
CONFIG_PREFIX = os.getenv("BLOB_CONFIG_PREFIX", "config")

def _get_bsc() -> BlobServiceClient:
    if CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(CONNECTION_STRING)
    if ACCOUNT_URL and SAS_TOKEN:
        token = SAS_TOKEN[1:] if SAS_TOKEN.startswith("?") else SAS_TOKEN
        return BlobServiceClient(account_url=ACCOUNT_URL, credential=token)
    raise RuntimeError("Blob 認証情報が不足しています（接続文字列 or URL+SAS を設定してください）。")

def _download_bytes(path_in_container: str) -> bytes:
    bsc = _get_bsc()
    bc = bsc.get_container_client(CONTAINER).get_blob_client(path_in_container)
    try:
        return bc.download_blob().readall()
    except ResourceNotFoundError:
        raise FileNotFoundError(f"Blob not found: {CONTAINER}/{path_in_container}")

def download_text(path_in_container: str, encoding="utf-8") -> str:
    """テキストファイルを取得"""
    return _download_bytes(path_in_container).decode(encoding)

def download_joblib_bytes(path_in_container: str) -> io.BytesIO:
    """joblibファイルをBytesIOで取得"""
    return io.BytesIO(_download_bytes(path_in_container))

def models_path(name: str) -> str:
    """モデルファイルのパスを生成"""
    return f"{MODELS_PREFIX.strip('/')}/{name}"

def config_path(name: str) -> str:
    """設定ファイルのパスを生成"""
    return f"{CONFIG_PREFIX.strip('/')}/{name}"
