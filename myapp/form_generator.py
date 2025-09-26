#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP重要度データから動的に入力フォームを生成するモジュール
"""

import json
import pandas as pd
from pathlib import Path
from django import forms
import os

def load_feature_analysis_data():
    """実際のCSVデータから特徴量の選択肢を分析して返す"""
    
    # CSVファイルのパス（環境変数から取得、なければデフォルト）
    csv_path = os.getenv('TRAINING_DATA_PATH', 
                        '/Users/kshintani/Documents/ソフトバンク/03_CI本部/KE/007_torihiki/app_code/取引審査データ_250829_train_data.csv')
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        # CSVが読めない場合のフォールバック
        return {}
    
    # 各特徴量の選択肢を分析
    feature_choices = {}
    
    # 貨物該非(1) - 分類選択
    if '貨物該非(1)' in df.columns:
        unique_vals = df['貨物該非(1)'].dropna().unique()
        feature_choices['貨物該非(1)'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 申請パターン - 分類選択
    if '申請パターン' in df.columns:
        unique_vals = df['申請パターン'].dropna().unique()
        feature_choices['申請パターン'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 物流/出荷パターン7 - 数値/テキスト
    if '物流/出荷パターン7' in df.columns:
        unique_vals = df['物流/出荷パターン7'].dropna().unique()
        if len(unique_vals) <= 50:  # 選択肢が少なければChoiceField
            feature_choices['物流/出荷パターン7'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 原産国(1) - 国名選択
    if '原産国(1)' in df.columns:
        unique_vals = df['原産国(1)'].dropna().unique()
        feature_choices['原産国(1)'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # EAR(1) - 分類選択
    if 'EAR(1)' in df.columns:
        unique_vals = df['EAR(1)'].dropna().unique()
        feature_choices['EAR(1)'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 商流/改正特一(圧力計)1 - 小さな分類
    if '商流/改正特一(圧力計)1' in df.columns:
        unique_vals = df['商流/改正特一(圧力計)1'].dropna().unique()
        feature_choices['商流/改正特一(圧力計)1'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # ECCN(1) - 分類選択
    if 'ECCN(1)' in df.columns:
        unique_vals = df['ECCN(1)'].dropna().unique()
        feature_choices['ECCN(1)'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 最終需要者(コード) - コード選択
    if '最終需要者(コード)' in df.columns:
        unique_vals = df['最終需要者(コード)'].dropna().unique()
        feature_choices['最終需要者(コード)'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 申請部署 - 部署選択
    if '申請部署' in df.columns:
        unique_vals = df['申請部署'].dropna().unique()
        feature_choices['申請部署'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 有償無償(1) - バイナリ選択
    if '有償無償(1)' in df.columns:
        unique_vals = df['有償無償(1)'].dropna().unique()
        feature_choices['有償無償(1)'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 商流/出荷パターン1 - パターン選択
    if '商流/出荷パターン1' in df.columns:
        unique_vals = df['商流/出荷パターン1'].dropna().unique()
        if len(unique_vals) <= 50:
            feature_choices['商流/出荷パターン1'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    # 取引目的 - 目的選択
    if '取引目的' in df.columns:
        unique_vals = df['取引目的'].dropna().unique()
        feature_choices['取引目的'] = sorted([str(v) for v in unique_vals if pd.notna(v)])
    
    return feature_choices

def get_top_features(top_k=10):
    """SHAP重要度から上位K個の特徴量を取得"""
    config_path = Path(__file__).parent.parent / 'config' / 'shap_feature_importance.json'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            shap_data = json.load(f)
        return [item[0] for item in shap_data['importance'][:top_k]]
    except Exception:
        # フォールバック: デフォルトの重要特徴量
        return [
            '貨物該非(1)', '申請パターン', '物流/出荷パターン7', '原産国(1)', 'EAR(1)',
            '商流/改正特一(圧力計)1', 'ECCN(1)', '最終需要者(コード)', '数量(1)', '品名(1)'
        ]

def create_dynamic_form_class(top_k=10):
    """SHAP重要度に基づいた動的フォームクラスを生成"""
    
    # 重要特徴量を取得
    top_features = get_top_features(top_k)
    
    # データから選択肢を取得
    feature_choices = load_feature_analysis_data()
    
    # フォームフィールドを動的に作成
    form_fields = {}
    
    for feature in top_features:
        field_name = feature.replace('(', '_').replace(')', '_').replace('/', '_')
        
        if feature in feature_choices:
            choices = feature_choices[feature]
            # 空文字選択肢を追加
            choices_with_empty = [('', '---選択してください---')] + [(c, c) for c in choices]
            
            if len(choices) <= 2:  # バイナリ選択
                form_fields[field_name] = forms.ChoiceField(
                    label=feature,
                    choices=choices_with_empty,
                    required=False,
                    widget=forms.RadioSelect
                )
            elif len(choices) <= 15:  # 小さな選択肢 - ドロップダウン
                form_fields[field_name] = forms.ChoiceField(
                    label=feature,
                    choices=choices_with_empty,
                    required=False
                )
            else:  # 大きな選択肢 - 検索可能なセレクト
                form_fields[field_name] = forms.ChoiceField(
                    label=feature,
                    choices=choices_with_empty,
                    required=False,
                    widget=forms.Select(attrs={'class': 'searchable-select'})
                )
        else:
            # 数値っぽいフィールドや自由テキスト
            if any(keyword in feature.lower() for keyword in ['数量', '単価', '金額']):
                form_fields[field_name] = forms.CharField(
                    label=feature,
                    required=False,
                    widget=forms.NumberInput(attrs={'placeholder': '数値を入力'})
                )
            else:
                form_fields[field_name] = forms.CharField(
                    label=feature,
                    required=False,
                    max_length=200,
                    widget=forms.TextInput(attrs={'placeholder': '入力してください'})
                )
    
    # 閾値とTop-K設定を追加
    form_fields['threshold'] = forms.FloatField(
        label='判定しきい値(0-1)',
        min_value=0.0,
        max_value=1.0,
        initial=0.5,
        required=False
    )
    
    form_fields['topk'] = forms.IntegerField(
        label='使用する特徴量の数（0=全て）',
        min_value=0,
        required=False,
        initial=top_k
    )
    
    # 動的にFormクラスを作成
    DynamicSHAPForm = type(
        'DynamicSHAPForm',
        (forms.Form,),
        form_fields
    )
    
    return DynamicSHAPForm

def get_field_mapping():
    """フォームフィールド名から元の特徴量名への変換マッピング"""
    top_features = get_top_features(10)
    mapping = {}
    
    for feature in top_features:
        field_name = feature.replace('(', '_').replace(')', '_').replace('/', '_')
        mapping[field_name] = feature
    
    return mapping

# デフォルトで上位10個の重要特徴量を使ったフォームを作成
TopSHAPForm = create_dynamic_form_class(top_k=10)