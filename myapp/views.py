# predictor/views.py
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from .forms import SinglePredictForm, convert_form_data
from .preprocess import preprocess_record, preprocess_df
from .model_service import predict_one, predict_df

def health_check(request):
    return HttpResponse("Hello, world. You're at the health check.")

def index(request):
    return render(request, 'myproject/index.html')

def _parse_topk(v):
    try:
        if v is None or v == "":
            return 0
        return max(0, int(v))
    except Exception:
        return 0

def single_input(request):
    ctx = {'form': SinglePredictForm()}
    if request.method == 'POST':
        form = SinglePredictForm(request.POST)
        if form.is_valid():
            # フォームデータを元の特徴量名に変換
            raw_data = form.cleaned_data.copy()
            data = convert_form_data(raw_data)
            
            threshold = float(data.pop('threshold', 0.5))
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
