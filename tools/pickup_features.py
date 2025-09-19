import json, numpy as np, pathlib

p = pathlib.Path("config/shap_feature_importance.json")
data = json.loads(p.read_text(encoding="utf-8"))["importance"]  # [["feat", score], ...]
# 念のため降順
data = sorted(data, key=lambda kv: kv[1], reverse=True)

names  = [k for k,_ in data]
scores = np.array([v for _,v in data], dtype=float)
scores = np.where(np.isfinite(scores), scores, 0.0)

total  = scores.sum()
share  = scores / total if total > 0 else np.zeros_like(scores)
cum    = share.cumsum()

out = [{"feature": n, "score": float(s), "share": float(sh), "cum_share": float(cs)}
       for n, s, sh, cs in zip(names, scores, share, cum)]

pathlib.Path("config/shap_cumulative.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)

# 目標累積で自動K（例：90%）
target = 0.80
K = int(np.searchsorted(cum, target) + 1)
print("K for >=90%:", K)
