import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Optional

def extract_insights(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    cat_col: Optional[str] = None,
    z_thresh: float = 2.5,
    max_insights: int = 6,
) -> List[Dict[str, Any]]:
    """Return top insights (trend/anomaly/segment). Keys match humor/charts."""
    out: List[Dict[str, Any]] = []
    nums = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    if not nums:
        return out

    out += find_trends(df, nums)
    out += find_anomalies(df, nums, z_thresh)
    out += find_segment_insights(df, cat_col)

    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    return out[:max_insights]

def calculate_percentage_change(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] == 0:
        return None
    return float(((s.iloc[-1] - s.iloc[0]) / abs(s.iloc[0])) * 100)

def find_trends(df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
    trends: List[Dict[str, Any]] = []
    x_full = np.arange(len(df))
    for col in numeric_cols:
        y = pd.to_numeric(df[col], errors="coerce")
        valid = ~y.isna()
        if valid.sum() < 3:
            continue
        slope, _, r, p, _ = stats.linregress(x_full[valid], y[valid].values)
        pct = calculate_percentage_change(y)
        if pct is None or abs(pct) < 5:  # 5% threshold
            continue
        direction = "increased" if pct > 0 else "decreased"
        trends.append({
            'type': 'trend',
            'metric': col,
            'value': pct,                      # pct-change
            'direction': direction,            # "increased"/"decreased"
            'summary': f"{col} {direction} by {abs(pct):.1f}% over the period",
            'score': abs(pct) * max(0.2, abs(r)),  # weight by correlation
            'detail': f"slope={slope:.4f}, r={r:.2f}, p={p:.4f}",
        })
    return trends

def find_anomalies(df: pd.DataFrame, numeric_cols: List[str], z_thresh: float) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue
        z = np.abs((s - s.mean()) / (s.std(ddof=0) + 1e-9))
        ridx = int(np.argmax(z))
        if z.iloc[ridx] >= z_thresh:
            abs_idx = int(s.index[ridx])
            anomalies.append({
                'type': 'anomaly',
                'metric': col,
                'value': float(s.iloc[ridx]),
                'z_score': float(z.iloc[ridx]),
                'index': abs_idx,  # absolute row index for chart marker
                'summary': f"Significant outlier in {col}: {s.iloc[ridx]:.2f} (z-score: {z.iloc[ridx]:.1f})",
                'score': float(z.iloc[ridx]) * 10.0,
            })
    return anomalies

def find_segment_insights(df: pd.DataFrame, cat_col: Optional[str]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    if cat_col is None:
        cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cats:
            return insights
        cat_col = cats[0]
    if df[cat_col].nunique() < 2 or df[cat_col].nunique() > 40:
        return insights
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return insights

    for num_col in num_cols:
        grp = df.groupby(cat_col)[num_col].mean()
        if len(grp) < 2:
            continue
        best = grp.idxmax(); worst = grp.idxmin()
        best_val, worst_val = grp.max(), grp.min()
        if best_val == 0:
            continue
        rel_gap = (best_val - worst_val) / abs(best_val)
        if rel_gap < 0.2:  # only surface if meaningful
            continue
        insights.append({
            'type': 'segment',
            'metric': f"{num_col} by {cat_col}",
            'best_segment': str(best),
            'best_value': float(best_val),
            'worst_segment': str(worst),
            'worst_value': float(worst_val),
            'summary': f"{best} leads in {num_col} with {best_val:.1f} (vs {worst}: {worst_val:.1f})",
            'score': float(rel_gap * 100),
        })
    return insights
