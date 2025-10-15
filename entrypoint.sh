#!/bin/bash

# Azure App Service での Django アプリケーション起動スクリプト

set -e

echo "Starting Django application..."

# データベースマイグレーション（必要に応じて）
echo "Running database migrations..."
python manage.py migrate --noinput

# 静的ファイルの収集（既にDockerfile内で実行済みだが、念のため）
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Django のセキュリティチェック
echo "Running security checks..."
python manage.py check --deploy

# Gunicorn でアプリケーションを起動
echo "Starting Gunicorn..."
exec gunicorn \
    --bind 0.0.0.0:8000 \
    --worker-class gevent \
    --worker-connections 1000 \
    --workers 2 \
    --timeout 300 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    myproject.wsgi:application