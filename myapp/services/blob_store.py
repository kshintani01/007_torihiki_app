import os
import io
from typing import Optional
from pathlib import Path
from django.conf import settings

# Azure Blob Storage認証情報（オプション）
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
ACCOUNT_URL = os.getenv("BLOB_ACCOUNT_URL")           # 例: https://<acct>.blob.core.windows.net
SAS_TOKEN = os.getenv("BLOB_SAS_TOKEN")               # 例: ?sv=...

CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "appdata")
MODELS_PREFIX = os.getenv("BLOB_MODELS_PREFIX", "models")
CONFIG_PREFIX = os.getenv("BLOB_CONFIG_PREFIX", "config")

def _get_bsc():
    """Azure Blob Service Client を取得（Azure接続が有効な場合のみ）"""
    try:
        from azure.storage.blob import BlobServiceClient
        if CONNECTION_STRING:
            return BlobServiceClient.from_connection_string(CONNECTION_STRING)
        if ACCOUNT_URL and SAS_TOKEN:
            token = SAS_TOKEN[1:] if SAS_TOKEN.startswith("?") else SAS_TOKEN
            return BlobServiceClient(account_url=ACCOUNT_URL, credential=token)
        raise RuntimeError("Blob 認証情報が不足しています（接続文字列 or URL+SAS を設定してください）。")
    except ImportError:
        raise RuntimeError("Azure Storage ライブラリがインストールされていません。")

def _download_from_azure(path_in_container: str) -> bytes:
    """Azure Blob Storage からバイナリデータをダウンロード"""
    from azure.core.exceptions import ResourceNotFoundError
    bsc = _get_bsc()
    bc = bsc.get_container_client(CONTAINER).get_blob_client(path_in_container)
    try:
        return bc.download_blob().readall()
    except ResourceNotFoundError:
        raise FileNotFoundError(f"Blob not found: {CONTAINER}/{path_in_container}")

def _download_from_local(file_path: str) -> bytes:
    """ローカルファイルシステムからバイナリデータを読み込み"""
    local_path = Path(settings.BASE_DIR) / file_path
    if not local_path.exists():
        raise FileNotFoundError(f"ローカルファイルが見つかりません: {local_path}")
    
    with open(local_path, 'rb') as f:
        return f.read()

def _download_bytes(path_in_container: str) -> bytes:
    """
    バイナリデータをダウンロード
    1. Azure Blob Storage を試す
    2. 失敗時はローカルファイルシステムにフォールバック
    """
    try:
        if CONNECTION_STRING or (ACCOUNT_URL and SAS_TOKEN):
            return _download_from_azure(path_in_container)
        else:
            # Azure接続情報が未設定の場合、直接ローカルを試す
            return _download_from_local(path_in_container)
    except Exception as azure_error:
        print(f"Azure Blob Storage からの読み込みに失敗: {azure_error}")
        try:
            return _download_from_local(path_in_container)
        except Exception as local_error:
            print(f"ローカルファイルからの読み込みに失敗: {local_error}")
            raise FileNotFoundError(f"ファイルが見つかりません: {path_in_container} (Azure: {azure_error}, Local: {local_error})")

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
