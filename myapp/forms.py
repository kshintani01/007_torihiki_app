from django import forms


class SinglePredictForm(forms.Form):
    # ダミーの3項目（型やバリデーションは後で差し替え可能）
    age = forms.IntegerField(label='年齢', min_value=0, max_value=120, required=False)
    income = forms.FloatField(label='年収(万円)', min_value=0, required=False)
    category = forms.ChoiceField(label='カテゴリ', choices=[('A', 'A'), ('B', 'B'), ('C', 'C')], required=False)
    threshold = forms.FloatField(label='判定しきい値(0-1)', min_value=0.0, max_value=1.0, initial=0.5)

class SinglePredictForm(forms.Form):
    age = forms.IntegerField(label='年齢', min_value=0, max_value=120, required=False)
    income = forms.FloatField(label='年収(万円)', min_value=0, required=False)
    category = forms.ChoiceField(label='カテゴリ', choices=[('A','A'),('B','B'),('C','C')], required=False)
    threshold = forms.FloatField(label='判定しきい値(0-1)', min_value=0.0, max_value=1.0, initial=0.5)
    # ★追加：Top-K（0または未入力=全特徴量使用）
    topk = forms.IntegerField(
        label='使用する特徴量の数（0=全て）',
        min_value=0, required=False, initial=0
    )