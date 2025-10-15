from pathlib import Path
import os
from django.core.management.utils import get_random_secret_key

# プロジェクトのベースディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = "h4cf#n=iz+#u*!bdaps2%andk!tu(o1lnc@w76ok=_nkfr4xx7"

# 環境変数からDEBUGモードを設定（デフォルトはFalse）
DEBUG = os.environ.get('DJANGO_DEBUG', 'False').lower() in ('true', '1', 'yes')

# 本番環境では環境変数からシークレットキーを取得
if not DEBUG:
    SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', SECRET_KEY)

# ALLOWED_HOSTSの設定（環境変数からも読み込み可能）
ALLOWED_HOSTS = [
    "sb-torihiki01.azurewebsites.net",
    "localhost", 
    "127.0.0.1",
]

# 環境変数からの追加ホスト設定
additional_hosts = os.environ.get('DJANGO_ALLOWED_HOSTS', '')
if additional_hosts:
    ALLOWED_HOSTS.extend([host.strip() for host in additional_hosts.split(',') if host.strip()])

# 開発環境では全ホストを許可
if DEBUG:
    ALLOWED_HOSTS.append("*")

# Azure App Service用のCSRF設定
CSRF_TRUSTED_ORIGINS = [
    "https://sb-torihiki01.azurewebsites.net",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# 環境変数からの追加CSRF信頼オリジン設定
additional_origins = os.environ.get('DJANGO_CSRF_TRUSTED_ORIGINS', '')
if additional_origins:
    CSRF_TRUSTED_ORIGINS.extend([origin.strip() for origin in additional_origins.split(',') if origin.strip()])

# Azure App Serviceのプロキシ設定
USE_X_FORWARDED_HOST = True
USE_X_FORWARDED_PORT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# CSRF設定の強化
CSRF_COOKIE_SECURE = not DEBUG  # HTTPSでのみCSRFクッキーを送信（本番環境）
CSRF_COOKIE_HTTPONLY = True     # JavaScriptからアクセス不可
CSRF_COOKIE_SAMESITE = 'Lax'    # CSRF攻撃を防ぐ


INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # 静的ファイル配信用
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


ROOT_URLCONF = 'myproject.urls'


TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'myapp' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
        'context_processors': [
            'django.template.context_processors.debug',
            'django.template.context_processors.request',
            'django.contrib.auth.context_processors.auth',
            'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


WSGI_APPLICATION = 'myproject.wsgi.application'


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'ja'
TIME_ZONE = 'Asia/Tokyo'
USE_I18N = True
USE_TZ = True


STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'  # Azure App Service用
STATICFILES_DIRS = [
    BASE_DIR / 'myapp' / 'static',
]

# WhiteNoise設定（静的ファイルの圧縮とキャッシュ）
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# セキュリティ設定の強化
if not DEBUG:
    # 本番環境でのセキュリティ設定
    SECURE_SSL_REDIRECT = True
    SECURE_HSTS_SECONDS = 31536000  # 1年
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = 'DENY'
    
    # セッション設定
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django.security': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}