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

def get_fallback_choices():
    """CSVが読めない場合のフォールバック選択肢"""
    return {
        '貨物該非(1)': [
            '該当', '対象外', '対象外(非該当)', '該当品あり', '非該当', 
            '該当 / 非該当', '該当(3点あり)', '全て非該当(Excel参照)', 
            '(該当品なし)', '-', '?象外'
        ],
        '申請パターン': [
            '①KSS起票/製品販売(社内CA該当)',
            '②KSS海外起票/部品販売(社内CA該当,EAR対象,該当品)',
            '④海外申請/現地クローズ',
            '⑤KE/海外Gr会社在庫使用',
            '⑥-2KE/該当品(EL顧客向け)',
            '⑥KE/該当品(KSECストック在庫を除く)',
            '⑦KE/該当品(KSECストック在庫)',
            '⑧KE/社内CA該当(EL顧客)',
            '⑨KE/社内CA該当(MEU顧客)',
            '⑪KE/役務',
            '⑮KE/出荷パターンC向け(EAR品＋非該当)'
        ],
        '物流/出荷パターン7': ['-', '0', '1', '2', '3', '4', '5', 'A', 'B', 'C'],
        '原産国(1)': [
            '日本', 'Japan', '米国', 'USA', '中国', '韓国', 'インドネシア', 
            'シンガポール', 'タイ', 'マレーシア', 'ドイツ', 'フランス', 
            'イタリア', 'オランダ', '台湾', 'インド', '米国以外', 
            '(米国製品なし)', '-'
        ],
        'EAR(1)': [
            '対象外', '対象', '該当', '米国', '規制対象外', 'KSEC発送のため該当',
            '対象外巣', '-'
        ],
        'EAR(2)': ['対象外', '対象', '規制対象外', '該当', '-'],
        'EAR(3)': ['対象外', '対象', '該当', '-'],
        'EAR(4)': ['対象外', '対象', '該当', '-'],
        'EAR(5)': ['対象外', '対象', '該当', '中国向けは対象外(米国成分25%未満)', '-'],
        'EAR(7)': ['対象外', '対象', '該当', '-'],
        'EAR(8)': ['対象外', '対象', '該当', '-'],
        'EAR(9)': ['対象外', '対象', '該当', '-'],
        'EAR(10)': ['対象外', '該当', '-'],
        '商流/改正特一(圧力計)1': ['-', 'E', 'a', '○', '可', '－'],
        'ECCN(1)': [
            'EAR99', '2B230', '2B999', '2B350', '3A999', '2B999,EAR99',
            '2B230,2B999,EAR99', '2B230,3A999,2B999,EAR99', '2B999/EAR99(ファイル参照)',
            '(KSEC)から出荷するためEAR99', '-', '--'
        ],
        '有償無償(1)': ['有償', '無償'],
        '商流/出荷パターン1': ['-', '0', '1', '2', '3', '4', '5', 'A', 'B', 'C'],
        '取引目的': [
            'BO/Back Orders',
            'KEKウェーハデモ/KEK Wafer Demonstration', 
            'その他/Others',
            'その他/Others（詳細は､(6)起票部署入力欄の起票部署コメントに記載してください）',
            '不具合対応/Defect handling',
            '技術提供/Technical Offerings',
            '海外生産（1年間包括審査）/Production outside Japan（one year including screening）',
            '海外生産（1年間包括審査／KEK）',
            '海外生産（単発審査）/Production outside Japan（Single-Step Screening）',
            '研修受入/Training Acceptance'
        ],
        '申請部署': [
            '(EA)', '(EAI)', '(EAII)', '(ES)', '(F企)', '(LBM)', '(ア営)',
            '(中営)', '(新素)', '(欧州)', '(米国)', '(韓国)', '(台湾)',
            '(中国)', '(東南)', '(シンガ)', '(インド)'
        ]
    }

def load_feature_analysis_data():
    """実際のCSVデータから特徴量の選択肢を分析して返す"""
    
    # 複数のCSVファイルパスを試行
    possible_paths = [
        # 環境変数で指定されたパス
        os.getenv('TRAINING_DATA_PATH'),
        # ローカル開発環境のパス
        '/Users/kshintani/Documents/ソフトバンク/03_CI本部/KE/007_torihiki/app_code/取引審査データ_250829_train_data.csv',
        # プロジェクトルートからの相対パス
        str(Path(__file__).parent.parent.parent / '取引審査データ_250829_train_data.csv'),
        # アプリ内の data フォルダ
        str(Path(__file__).parent.parent / 'data' / 'train_data.csv'),
    ]
    
    # 有効なパスを探して読み込み
    for csv_path in possible_paths:
        if csv_path and Path(csv_path).exists():
            try:
                print(f"CSVファイルを読み込み中: {csv_path}")
                df = pd.read_csv(csv_path, low_memory=False)
                break
            except Exception as e:
                print(f"CSVファイル読み込みエラー（{csv_path}): {e}")
                continue
    else:
        print("有効なCSVファイルが見つかりません。フォールバック選択肢を使用します。")
        # CSVが読めない場合のフォールバック：重要特徴量の既知の選択肢
        return get_fallback_choices()
    
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
        # ダッシュのバリエーションを統一
        normalized_vals = []
        for v in unique_vals:
            if pd.notna(v):
                str_v = str(v)
                # 複数種類のダッシュを単一の"-"に統一
                if str_v in ['--', '―', 'ー']:
                    str_v = '-'
                normalized_vals.append(str_v)
        feature_choices['EAR(1)'] = sorted(list(set(normalized_vals)))
    
    # EAR(2-5, 7-10) - 分類選択
    for ear_num in [2, 3, 4, 5, 7, 8, 9, 10]:
        ear_col = f'EAR({ear_num})'
        if ear_col in df.columns:
            unique_vals = df[ear_col].dropna().unique()
            # ダッシュのバリエーションを統一
            normalized_vals = []
            for v in unique_vals:
                if pd.notna(v):
                    str_v = str(v)
                    # 複数種類のダッシュを単一の"-"に統一
                    if str_v in ['--', '―', 'ー']:
                        str_v = '-'
                    normalized_vals.append(str_v)
            feature_choices[ear_col] = sorted(list(set(normalized_vals)))
    
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
    
    # データが取得できなかった場合のフォールバック
    if not feature_choices:
        print("CSVからの特徴量分析に失敗。フォールバック選択肢を使用します。")
        return get_fallback_choices()
    
    print(f"CSVから{len(feature_choices)}個の特徴量の選択肢を取得しました。")
    return feature_choices

def get_top_features(top_k=10):
    """SHAP重要度から上位K個の特徴量を取得し、CSV列順でソート"""
    config_path = Path(__file__).parent.parent / 'config' / 'shap_feature_importance.json'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            shap_data = json.load(f)
        top_features = [item[0] for item in shap_data['importance'][:top_k]]
        
        # CSV列順を取得
        csv_column_order = get_csv_column_order()
        
        # SHAP上位特徴量をCSV列順でソート
        ordered_features = []
        for col in csv_column_order:
            if col in top_features:
                ordered_features.append(col)
        
        # もしCSV列順で全部埋まらなかった場合は残りを追加
        for feature in top_features:
            if feature not in ordered_features:
                ordered_features.append(feature)
                
        return ordered_features[:top_k]
        
    except Exception:
        # フォールバック: CSV列順での重要特徴量
        return get_fallback_feature_order()

def get_csv_column_order():
    """CSVファイルから列の順序を取得"""
    possible_paths = [
        os.getenv('TRAINING_DATA_PATH'),
        '/Users/kshintani/Documents/ソフトバンク/03_CI本部/KE/007_torihiki/app_code/取引審査データ_250829_train_data.csv',
        str(Path(__file__).parent.parent.parent / '取引審査データ_250829_train_data.csv'),
        str(Path(__file__).parent.parent / 'data' / 'train_data.csv'),
    ]
    
    for csv_path in possible_paths:
        if csv_path and Path(csv_path).exists():
            try:
                # 列名のみ取得（最初の行だけ読む）
                df = pd.read_csv(csv_path, nrows=0)
                return df.columns.tolist()
            except Exception:
                continue
    
    return get_fallback_feature_order()

def get_fallback_feature_order():
    """CSVが読めない場合のフォールバック特徴量順序（想定されるCSV列順）"""
    return [
        '貨物該非(1)', '申請パターン', '物流/出荷パターン7', '原産国(1)', 'EAR(1)',
        '商流/改正特一(圧力計)1', 'ECCN(1)', '有償無償(1)', '商流/出荷パターン1', '取引目的'
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
        elif feature in get_fallback_choices():
            # フォールバック選択肢を使用
            choices = get_fallback_choices()[feature]
            choices_with_empty = [('', '---選択してください---')] + [(c, c) for c in choices]
            
            if len(choices) <= 2:
                form_fields[field_name] = forms.ChoiceField(
                    label=feature,
                    choices=choices_with_empty,
                    required=False,
                    widget=forms.RadioSelect
                )
            elif len(choices) <= 15:
                form_fields[field_name] = forms.ChoiceField(
                    label=feature,
                    choices=choices_with_empty,
                    required=False
                )
            else:
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
    
    # Top-K設定を追加（閾値は内部的に0固定）
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