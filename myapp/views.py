# predictor/views.py
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from .forms import SinglePredictForm, convert_form_data
from .preprocess import preprocess_record, preprocess_df
from .model_service import predict_one, predict_df

def health_check(request):
    return HttpResponse("Hello, world. You're at the health check.")

def debug_form(request):
    """フォーム生成デバッグ用エンドポイント"""
    from .form_generator import load_feature_analysis_data, get_top_features, get_fallback_choices
    import json
    
    debug_info = {
        'top_features': get_top_features(10),
        'feature_choices_count': 0,
        'fallback_choices_count': len(get_fallback_choices()),
        'form_fields_count': 0,
        'errors': []
    }
    
    try:
        feature_choices = load_feature_analysis_data()
        debug_info['feature_choices_count'] = len(feature_choices)
        debug_info['feature_choices_keys'] = list(feature_choices.keys())
    except Exception as e:
        debug_info['errors'].append(f'load_feature_analysis_data error: {str(e)}')
    
    try:
        from .forms import SinglePredictForm
        form = SinglePredictForm()
        debug_info['form_fields_count'] = len(form.fields)
        debug_info['form_field_types'] = {
            name: type(field).__name__ for name, field in form.fields.items()
        }
    except Exception as e:
        debug_info['errors'].append(f'Form creation error: {str(e)}')
    
    return HttpResponse(
        f"<pre>{json.dumps(debug_info, indent=2, ensure_ascii=False)}</pre>", 
        content_type='text/html; charset=utf-8'
    )

def index(request):
    return render(request, 'myproject/index.html')

def _parse_topk(v):
    try:
        if v is None or v == "":
            return 0
        return max(0, int(v))
    except Exception:
        return 0
    
def _parse_threshold(v):
    """空欄・不正値を許容しつつ 0–1 に収める"""
    try:
        if v is None or v == "":
            return 0.5
        x = float(v)
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x
    except Exception:
        return 0.5

def single_input(request):
    ctx = {'form': SinglePredictForm()}
    if request.method == 'POST':
        form = SinglePredictForm(request.POST)
        if form.is_valid():
            # フォームデータを元の特徴量名に変換
            raw_data = form.cleaned_data.copy()
            data = convert_form_data(raw_data)
            
            threshold = _parse_threshold(data.pop('threshold', None))
            topk = _parse_topk(data.pop('topk', 0))
            
            # 空文字列やNoneを適切にハンドリング
            cleaned_data = {}
            for key, value in data.items():
                if value is not None and value != '':
                    cleaned_data[key] = value
            
            X = preprocess_record(cleaned_data).to_dict(orient='records')[0]
            result = predict_one(X, threshold, topk=topk)
            ctx.update({
                'form': form, 
                'result': result, 
                'features': X,
                'input_data': cleaned_data  # 入力データも表示用に保持
            })
        else:
            ctx['form'] = form
    return render(request, 'myproject/single.html', ctx)

def csv_upload(request):
    ctx = {}
    if request.method == 'POST' and request.FILES.get('csv_file'):
        f = request.FILES['csv_file']
        topk = _parse_topk(request.POST.get('topk'))
        try:
            df = pd.read_csv(f)
        except Exception as e:
            ctx['error'] = f'CSV読込に失敗しました: {e}'
            return render(request, 'myproject/csv.html', ctx)
        X = preprocess_df(df)
        out = predict_df(X, topk=topk)
        csv_bytes = out.to_csv(index=False).encode('utf-8-sig')
        response = HttpResponse(csv_bytes, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="scored.csv"'
        return response
    return render(request, 'myproject/csv.html', ctx)
