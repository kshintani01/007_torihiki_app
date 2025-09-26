from django import forms
from .form_generator import TopSHAPForm, get_field_mapping

# SHAP重要度に基づく動的フォーム
class SinglePredictForm(TopSHAPForm):
    """
    SHAP重要度上位の特徴量を動的に生成したフォーム
    """
    pass

# フィールド名変換用の関数
def convert_form_data(form_data):
    """
    フォームデータのフィールド名を元の特徴量名に変換
    """
    mapping = get_field_mapping()
    converted = {}
    
    for field_name, value in form_data.items():
        if field_name in mapping:
            # 元の特徴量名に変換
            original_name = mapping[field_name]
            converted[original_name] = value
        else:
            # 変換不要（threshold, topkなど）
            converted[field_name] = value
    
    return converted

# 後方互換用の従来フォーム（必要に応じて使用）
class LegacySinglePredictForm(forms.Form):
    age = forms.IntegerField(label='年齢', min_value=0, max_value=120, required=False)
    income = forms.FloatField(label='年収(万円)', min_value=0, required=False)
    category = forms.ChoiceField(label='カテゴリ', choices=[('A','A'),('B','B'),('C','C')], required=False)
    threshold = forms.FloatField(label='判定しきい値(0-1)', min_value=0.0, max_value=1.0, initial=0.5)
    topk = forms.IntegerField(
        label='使用する特徴量の数（0=全て）',
        min_value=0, required=False, initial=0
    )