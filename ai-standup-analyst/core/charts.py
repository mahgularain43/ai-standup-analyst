import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def create_chart(df: pd.DataFrame, insight: Dict[str, Any], date_col: Optional[str] = None, rolling: int = 7):
    t = insight['type']
    metric = insight['metric']
    if t == 'trend':
        return _trend_chart(df, metric, date_col, rolling)
    elif t == 'anomaly':
        return _anomaly_chart(df, metric, insight, date_col)
    elif t == 'segment':
        return _segment_chart(df, insight)
    else:
        return _basic_chart(df, metric, date_col)

def _x(df: pd.DataFrame, date_col: Optional[str]):
    if date_col and date_col in df.columns:
        return df[date_col]
    return pd.Series(range(len(df)), name="Index")

def _trend_chart(df, metric, date_col, rolling):
    x = _x(df, date_col)
    s = pd.to_numeric(df[metric], errors="coerce")
    fig = go.Figure()
    fig.add_scatter(x=x, y=s, mode="lines+markers", name=metric)
    if len(s.dropna()) >= rolling:
        fig.add_scatter(
            x=x, y=s.rolling(rolling).mean(), mode="lines",
            name=f"Rolling mean ({rolling})", line=dict(dash="dash")
        )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def _anomaly_chart(df, metric, insight, date_col):
    x = _x(df, date_col)
    s = pd.to_numeric(df[metric], errors="coerce")
    fig = go.Figure()
    fig.add_scatter(x=x, y=s, mode="lines+markers", name=metric)
    idx = insight.get("index", None)
    if idx is not None and 0 <= idx < len(df):
        fig.add_scatter(x=[x.iloc[idx]], y=[s.iloc[idx]], mode="markers",
                        marker=dict(size=14, symbol="star"), name="Anomaly")
        fig.add_vline(x=x.iloc[idx], line_width=1, line_dash="dot", opacity=0.4)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def _segment_chart(df, insight):
    # metric looks like "num by cat"
    parts = insight['metric'].split(' by ')
    if len(parts) != 2:
        return _basic_chart(df, parts[0] if parts else df.columns[0], None)
    num_col, cat_col = parts
    grouped = df.groupby(cat_col)[num_col].mean().reset_index()
    fig = px.bar(grouped, x=cat_col, y=num_col, title=f"Segment Analysis: {num_col} by {cat_col}")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig

def _basic_chart(df, metric, date_col):
    x = _x(df, date_col)
    s = pd.to_numeric(df[metric], errors="coerce")
    fig = px.area(x=x, y=s, labels={"x": x.name or "Index", "y": metric}, title=f"Data Overview: {metric}")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig
