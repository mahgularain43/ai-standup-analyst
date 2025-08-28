import os, io, time, random, math, json
from datetime import timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests  # for Slack webhook

from core.insights import extract_insights
from core.humor import make_jokes, PERSONA_PRESETS
from core.charts import create_chart
from core.utils import generate_pdf_report, create_sample_data
from core.voice import speak_to_file, add_laugh_track


# ==============================
# App & Global Styles
# ==============================
st.set_page_config(
    page_title="AI Stand-Up Analyst 🎤",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_CSS = """
<style>
:root{
  --bg:#ffffff;
  --card:#ffffff;
  --muted:#6b7280;
  --primary:#8E0F18;
  --accent:#1f2937;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
h1,h2,h3,h4 { color: var(--accent); }
.small-muted{ color: var(--muted); font-size:.9rem; }
.chip{display:inline-block;padding:.25rem .6rem;border-radius:999px;border:1px solid #e5e7eb;background:#f9fafb;margin-right:.5rem;font-size:.85rem}
.section-card{
  border:1px solid #eee; border-radius:14px; padding:16px; background:var(--card);
  box-shadow:0 2px 12px rgba(0,0,0,.05); margin-bottom:14px;
}
.main-header{
  text-align:center; padding:1.1rem 0; font-size:clamp(1.8rem,3.5vw,2.6rem);
  font-weight:800; letter-spacing:.2px;
  background:linear-gradient(90deg,var(--primary),#141414);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.pill{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .65rem;border-radius:999px;border:1px solid #e5e7eb;font-size:.85rem}
.pill-dot{width:.6rem;height:.6rem;border-radius:50%;}
.insight-card{border:1px solid #eee;border-radius:12px;padding:14px 16px;background:#fff;box-shadow:0 3px 16px rgba(0,0,0,.06);}
.footer-note{color:#6b7280;font-size:.85rem}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown('<div class="main-header">AI Stand-Up Analyst 🤖🎤</div>', unsafe_allow_html=True)
st.caption("Turn your data into comedy gold — with real insights attached.")


# ==============================
# Helpers
# ==============================
@st.cache_data(show_spinner=False)
def _read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def _guess_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time")):
            return c
    return None

def _spice_score(insights: list) -> float:
    s = 0.0
    for ins in insights:
        t = ins.get("type")
        if t == "trend":
            s += min(100, abs(ins.get("value", 0))) * 0.6
        elif t == "anomaly":
            s += min(10, ins.get("z_score", 0)) * 5.0
        elif t == "segment":
            try:
                spread = abs(ins["best_value"] - ins["worst_value"]) / max(1e-9, abs(ins["best_value"]))
            except Exception:
                spread = 0
            s += min(100, spread * 100) * 0.25
    return float(max(0, min(100, s)))

def _infer_mood(insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    score = 0.0
    trend_count = 0
    for ins in insights:
        if ins["type"] == "trend":
            trend_count += 1
            val = ins.get("value", 0.0)
            score += (1 if val > 0 else -1) * min(1.0, abs(val) / 25.0)
        elif ins["type"] == "anomaly":
            score -= min(0.5, ins.get("z_score", 0.0) / 10.0)

    if trend_count == 0:
        label, color = "confused", "#f59e0b"
    else:
        if score > 0.4: label, color = "optimistic", "#10b981"
        elif score < -0.4: label, color = "pessimistic", "#ef4444"
        else: label, color = "mixed", "#f59e0b"

    return {"label": label, "color": color, "score": round(score, 2)}

def _apply_dynamic_theme(mood: Dict[str, Any]):
    tint = "#f0fdf4" if mood["label"] == "optimistic" else ("#fff7ed" if mood["label"] in ("mixed", "confused") else "#fef2f2")
    st.markdown(f"<style>:root {{ --bg:{tint}; }}</style>", unsafe_allow_html=True)

def _primary_metric(metrics: List[str]) -> Optional[str]:
    return metrics[0] if metrics else None

def _linear_forecast(df: pd.DataFrame, metric: str, date_col: Optional[str], horizon: int = 14):
    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    if s.empty: return None, None
    if date_col and date_col in df.columns:
        x = pd.to_datetime(df[date_col], errors="coerce").dropna()
        x = (x - x.min()).dt.days.loc[s.index].values
    else:
        x = np.arange(len(s))
    a, b = np.polyfit(x, s.values, 1)
    fit_y = a * x + b
    last_x = x[-1]
    future_x = np.arange(last_x + 1, last_x + horizon + 1)
    future_y = a * future_x + b
    if date_col and date_col in df.columns:
        last_date = pd.to_datetime(df[date_col].iloc[-1], errors="coerce")
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    else:
        future_dates = list(range(len(s), len(s) + horizon))
    return (x, s.values, fit_y), (future_dates, future_y, a, b)

def _forecast_story(metric: str, slope: float, future_dates: List, future_vals: np.ndarray) -> str:
    direction = "up" if slope > 0 else "down"
    mid_idx = min(6, len(future_vals)-1)
    mid_val = future_vals[mid_idx]
    mid_when = future_dates[mid_idx]
    when_str = str(pd.to_datetime(mid_when).date()) if isinstance(mid_when, (pd.Timestamp, np.datetime64)) else f"t+{mid_idx+1}"
    return (f"Projection suggests **{metric}** is trending **{direction}**. "
            f"Around **{when_str}**, you might be near **{mid_val:.1f}**. "
            "If this rate holds, prep your slides: the future has punchlines.")

# Mood Ring
def _mood_ring(score: float, label: str, color: str):
    val = max(0, min(100, int((score + 1) * 50)))
    fig = go.Figure(go.Pie(values=[val, 100 - val], hole=0.7, showlegend=False))
    fig.update_traces(textinfo='none', marker=dict(colors=[color, '#e5e7eb']))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=140,
                      annotations=[dict(text=label.title(), showarrow=False, font=dict(size=12))])
    return fig

# Scenario Planning
def _scenario_stories(metric: str, slope: float, pct: float = 0.3):
    best_slope = slope * (1 + pct)
    worst_slope = slope * (1 - pct)
    return {
        "best": (best_slope, f"Best case: **{metric}** accelerates — queue victory lap playlist."),
        "worst": (worst_slope, f"Worst case: **{metric}** slows — supportive tone, tighter execution."),
    }

# Crisis wrapper
def _crisis_wrap(joke: str, mood_label: str) -> str:
    return f"💡 Heads up: {joke}\n(We’ve got this. Action > Angst.)" if mood_label == "pessimistic" else joke

# Industry tails
INDUSTRY_TAILS = {
    "General": "",
    "E-commerce": " — cart’s feeling needy; nudge with a discount?",
    "Fintech": " — volatility? Or just my coffee intake?",
    "Healthcare": " — prescribe a check on sample bias.",
}

# Data health
def _health_check(df: pd.DataFrame, date_col_guess: str | None):
    issues = []
    if df.empty:
        issues.append("Dataset is empty.")
        return issues
    null_share = df.isna().mean(numeric_only=False).sort_values(ascending=False)
    high_null = null_share[null_share > 0.2]
    if not high_null.empty:
        cols = ", ".join([f"{c} ({p:.0%})" for c, p in high_null.items()])
        issues.append(f"High missing values in: {cols}")
    dups = df.duplicated().sum()
    if dups: issues.append(f"{dups} duplicated rows found.")
    if date_col_guess and date_col_guess in df.columns:
        try:
            _ = pd.to_datetime(df[date_col_guess], errors="raise")
        except Exception:
            issues.append(f'Column "{date_col_guess}" looks like a date but failed to parse.')
    if df.select_dtypes(include="number").shape[1] == 0:
        issues.append("No numeric columns detected — select metrics manually or check data types.")
    return issues

# Quiz
def _quiz_from_insight(ins: Dict[str, Any]):
    metric = ins.get("metric", "the metric")
    q = f"What did we notice about **{metric}**?"
    a = ins["summary"]
    wrong = [f"{metric} stayed flat all period",
             f"Seasonality dominated with no anomalies",
             f"Segment differences were minimal"]
    opts = [a] + wrong
    random.shuffle(opts)
    return q, opts, a

# A/B hint
def _ab_test_hint(df: pd.DataFrame, metric: Optional[str], group_col: Optional[str]):
    if not metric or not group_col or group_col not in df.columns: return None
    try:
        g = df.dropna(subset=[metric, group_col]).groupby(group_col)[metric].mean().sort_values(ascending=False)
    except Exception:
        return None
    if len(g) < 2: return None
    a, b = g.index[:2]
    base = abs(g.iloc[1]) + 1e-9
    lift = (g.iloc[0] - g.iloc[1]) / base * 100
    if abs(lift) < 5: return None
    return f"A/B vibe: **{a}** beats **{b}** by {lift:.1f}%. Group {b} is requesting a rematch."

# Achievements
def _achievements(insights: List[Dict[str, Any]], jokes_generated: int) -> List[str]:
    badges = []
    if len(insights) >= 6: badges.append("🏅 Trend Spotter")
    if any(i["type"] == "anomaly" and i.get("z_score", 0) >= 3 for i in insights): badges.append("🎯 Anomaly Hunter")
    if jokes_generated >= 10: badges.append("🥇 Comedy Gold")
    return badges

# Slack
def _post_slack(webhook_url: str, text: str) -> bool:
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=5)
        return r.status_code < 300
    except Exception:
        return False


# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("🎛️ Controls")

# Data loading
st.sidebar.subheader("📤 Data")
uploaded = st.sidebar.file_uploader(
    "Upload data file",
    type=["csv", "xlsx", "xls", "parquet"],
    help="CSV, Excel (.xlsx/.xls) or Parquet. Max ~200MB."
)

def _load_any(file):
    name = getattr(file, "name", "uploaded")
    ext = os.path.splitext(name.lower())[1]
    try:
        if ext == ".csv": return pd.read_csv(file)
        elif ext in (".xlsx", ".xls"): return pd.read_excel(file)
        elif ext == ".parquet": return pd.read_parquet(file)
        else:
            st.warning("Unsupported file type. Please upload CSV, Excel, or Parquet.")
            return None
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

lc, rc = st.sidebar.columns([1, 1])
if lc.button("📦 Load Sample"):
    st.session_state["df"] = create_sample_data()
if rc.button("⬇️ Download template CSV"):
    _tmpl = create_sample_data(12)[["date","sales","customers","region","product"]]
    _tmpl.to_csv("data_template.csv", index=False)
    with open("data_template.csv", "rb") as f:
        st.sidebar.download_button("Get data_template.csv", f, file_name="data_template.csv", mime="text/csv")

if uploaded:
    df_new = _load_any(uploaded)
    if df_new is not None and not df_new.empty:
        st.session_state["df"] = df_new

df = st.session_state.get("df")
if df is None:
    st.info("Upload a file or click **Load Sample**.")
    st.stop()

# Time & resampling
st.sidebar.subheader("⏱️ Time & Resampling")
date_guess = _guess_date_col(df)
st.sidebar.caption("If your file has a date/time column, select it for time-aware charts.")
date_col = st.sidebar.selectbox("Date column (optional)", [None] + list(df.columns),
                                index=(0 if not date_guess else 1 + list(df.columns).index(date_guess)))
freq = st.sidebar.selectbox("Resample", ["None", "Week", "Month"], index=0)
st.sidebar.caption("Resample aggregates by week or month if your data is daily or event-level.")
agg = st.sidebar.selectbox("Aggregate", ["mean", "sum"], index=0)

# Metrics & segment
st.sidebar.subheader("📈 Select metrics to analyze")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
metrics = st.sidebar.multiselect("Numeric metrics", numeric_cols, default=numeric_cols[:2] if numeric_cols else [])
segment_col = st.sidebar.selectbox("Segment by (optional)", [None] + cat_cols, index=0)

# Detection
st.sidebar.subheader("🧪 Detection")
z_thresh = st.sidebar.slider("Anomaly z-threshold", 2.0, 4.0, 2.5, 0.1)
rolling = st.sidebar.slider("Rolling window (charts)", 3, 21, 7, 1)
max_insights = st.sidebar.slider("Max insights", 3, 12, 6)

# Persona
st.sidebar.subheader("🎭 Persona")
persona = st.sidebar.selectbox("Comedian persona", list(PERSONA_PRESETS.keys()), index=0)
st.sidebar.caption(PERSONA_PRESETS[persona]["description"])
base_tone = st.sidebar.slider("Comedy intensity", 1, 10, 6)

# Industry
st.sidebar.subheader("🌎 Industry flavor")
industry = st.sidebar.selectbox("Pick an industry tone", list(INDUSTRY_TAILS.keys()), index=0)

# Voice
st.sidebar.subheader("🗣️ Voice Assistant")
voice_name = st.sidebar.text_input("Voice name (optional)", value="")
laugh_track = st.sidebar.checkbox("Add laugh track", value=False)
typewriter = st.sidebar.checkbox("Typewriter effect", value=True)

# State
if "regen_nonce" not in st.session_state: st.session_state["regen_nonce"] = 0
if "tone_offsets" not in st.session_state: st.session_state["tone_offsets"] = {}
if "jokes_count" not in st.session_state: st.session_state["jokes_count"] = 0
if "quiz_answers" not in st.session_state: st.session_state["quiz_answers"] = {}
if "joke_fb" not in st.session_state: st.session_state["joke_fb"] = []  # feedback tracker
if "slack_url" not in st.session_state: st.session_state["slack_url"] = ""


# ==============================
# Preprocess & Insights
# ==============================
work = df.copy()
if date_col:
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

if date_col and freq != "None":
    rule = {"Week": "W", "Month": "M"}[freq]
    agg_fn = "mean" if agg == "mean" else "sum"
    work = work.set_index(date_col).resample(rule).agg(agg_fn).reset_index()

if not metrics:
    st.error("Please select at least one numeric metric.")
    st.stop()

with st.expander("🩺 Data health check", expanded=False):
    hc = _health_check(df, _guess_date_col(df))
    if hc: [st.warning(i) for i in hc]
    else: st.success("No obvious issues detected.")

insights = extract_insights(
    work, numeric_cols=metrics, cat_col=segment_col, z_thresh=z_thresh, max_insights=max_insights,
)
if not insights:
    st.warning("No insights found. Try different metrics, lower the z-threshold, or resample.")
    st.stop()

# Mood & theme
mood = _infer_mood(insights)
_apply_dynamic_theme(mood)
if mood["label"] == "pessimistic":
    persona = "Executive Coach"
    base_tone = min(base_tone, 7)

# Breaking banner
top_ano = next((i for i in insights if i["type"] == "anomaly" and i.get("z_score", 0) >= 3), None)
if top_ano: st.warning(f"🗞️ Breaking: {top_ano['summary']}")


# ==============================
# Top Overview
# ==============================
top_left, top_mid, top_right = st.columns([1.0, 1.2, 1.2])
with top_left:
    st.markdown("#### Mood")
    st.markdown(
        f'<div class="pill"><span class="pill-dot" style="background:{mood["color"]}"></span>'
        f'{mood["label"].title()} <span class="small-muted">score {mood["score"]:+.2f}</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_mood_ring(mood["score"], mood["label"], mood["color"]), use_container_width=True)

with top_mid:
    st.markdown("#### Roast & Stats")
    spice_val = _spice_score(insights)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Roast Meter", f"{spice_val:.1f}")
    c2.metric("Trends", sum(1 for i in insights if i["type"] == "trend"))
    c3.metric("Anomalies", sum(1 for i in insights if i["type"] == "anomaly"))
    c4.metric("Segments", sum(1 for i in insights if i["type"] == "segment"))
    st.caption("Tabs: **Insights**, **Create**, **Quiz**, **Forecast**, **Export**, **Settings**.")

with top_right:
    st.markdown("#### Recommendations")
    def _recommendations(ins: List[Dict[str, Any]], mets: List[str]) -> List[str]:
        recs = []
        if any(i["type"] == "anomaly" for i in ins):
            recs.append("Investigate anomalies: check deployments, data quality, and segmentation.")
        if any(i["type"] == "trend" and abs(i.get("value", 0)) > 15 for i in ins):
            recs.append("High-magnitude trend — alert stakeholders or schedule a deep dive.")
        if "customers" in [m.lower() for m in mets]:
            recs.append("Consider cohort analysis for churn/retention.")
        if not recs:
            recs.append("Try switching persona or raising comedy intensity to vary punchlines.")
        return recs
    [st.markdown(f"• {r}") for r in _recommendations(insights, metrics)]

st.markdown("---")


# ==============================
# Tabs
# ==============================
tab1, tab2, tab_quiz, tab3, tab4, tab5 = st.tabs(
    ["📌 Insights & Jokes", "🎨 Create", "🧩 Quiz", "🔮 Forecast", "📄 Export & Share", "⚙️ Settings"]
)

# ---- Tab 1: Insights & Jokes ----
with tab1:
    st.markdown("### Highlights")
    figs_for_pdf = []

    # A/B hint
    ab_hint = _ab_test_hint(work, _primary_metric(metrics), segment_col)
    if ab_hint: st.info(ab_hint)

    for idx, ins in enumerate(insights):
        key = f"{ins['type']}::{ins['summary']}"
        tone_offset = st.session_state["tone_offsets"].get(key, 0)

        with st.container():
            colA, colB = st.columns([2.2, 1.2])

            with colA:
                fig = create_chart(work, ins, date_col=date_col, rolling=rolling)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    figs_for_pdf.append((ins["summary"], fig))

            with colB:
                seed_val = hash(key) + st.session_state["regen_nonce"]
                random.seed(seed_val)
                joke = make_jokes([ins], persona=persona,
                                  tone=max(1, min(10, base_tone + tone_offset)))[0]
                joke = _crisis_wrap(joke, mood["label"]) + INDUSTRY_TAILS.get(industry, "")

                st.markdown(f"**{ins['summary']}**")
                if ins.get("detail"): st.caption(ins["detail"])

                if typewriter:
                    ph = st.empty()
                    txt = ""
                    full = f'“{joke}”'
                    for ch in full:
                        txt += ch
                        ph.markdown(
                            f'<div class="insight-card"><h4>🎤 Punchline</h4><p><em>{txt}</em></p></div>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.01)
                else:
                    st.markdown(
                        f'<div class="insight-card"><h4>🎤 Punchline</h4><p><em>“{joke}”</em></p></div>',
                        unsafe_allow_html=True,
                    )

                # feedback + controls
                t = st.slider("Tone tweak", -3, 3, tone_offset, key=f"tone_{idx}")
                st.session_state["tone_offsets"][key] = t

                c1, c2, c3, c4 = st.columns([1,1,1,1])
                with c1:
                    if st.button("↻ Regen", key=f"regen_{idx}"):
                        st.session_state["regen_nonce"] += 1
                        st.rerun()
                with c2:
                    if st.button("🎙️ Perform", key=f"perform_{idx}"):
                        audio_basename = f"joke_{idx}.wav"
                        audio_path = speak_to_file(joke, audio_basename, voice_name=voice_name)
                        final_path = audio_path
                        if laugh_track and audio_path and audio_path.endswith((".wav", ".mp3")):
                            out_mix = f"joke_{idx}_mix.wav"
                            mixed = add_laugh_track(audio_path, out_mix, intensity=0.6)
                            if mixed: final_path = mixed
                        if final_path and os.path.exists(final_path): st.audio(final_path)
                        else: st.info("Audio not available (TTS engine missing).")
                with c3:
                    if st.button("👍", key=f"up_{idx}"):
                        st.session_state["joke_fb"].append({"summary": ins["summary"], "joke": joke, "vote": 1})
                with c4:
                    if st.button("👎", key=f"dn_{idx}"):
                        st.session_state["joke_fb"].append({"summary": ins["summary"], "joke": joke, "vote": -1})

                st.session_state["jokes_count"] += 1

    # Achievements
    badges = _achievements(insights, st.session_state["jokes_count"])
    if badges:
        st.markdown("### Achievements")
        st.markdown(" ".join([f'<span class="chip">{b}</span>' for b in badges]), unsafe_allow_html=True)

    # Slack notify button (top anomaly)
    st.markdown("### Alerts")
    if st.button("Notify Slack about top anomaly"):
        if not st.session_state["slack_url"]:
            st.warning("Add your Slack Incoming Webhook URL in **Settings**.")
        else:
            ano = next((i for i in insights if i["type"] == "anomaly"), None)
            if ano:
                msg = f":rotating_light: Anomaly in *{ano['metric']}*: {ano['summary']}"
                ok = _post_slack(st.session_state["slack_url"], msg)
                st.success("Posted to Slack!") if ok else st.error("Slack post failed.")
            else:
                st.info("No anomaly right now.")

# ---- Tab 2: Create ----
with tab2:
    st.markdown("### Creative Generators")
    left, right = st.columns([1.3, 1.7])

    def _haiku(ins: List[Dict[str, Any]]) -> str:
        line1 = "numbers softly rise"
        line2 = "charts whisper secret stories"
        line3 = "punchlines find the truth"
        for it in ins:
            if it["type"] == "trend":
                line1 = "lines climb patiently" if it.get("value", 0) > 0 else "lines bow to gravity"
                break
        return f"{line1}\n{line2}\n{line3}"

    def _rap(ins: List[Dict[str, Any]]) -> str:
        bars = [
            "Yo, check the axis, we flexin' on the average,",
            "Trends so clean, outliers need a bandage.",
            "Segments in the mix, North side with the leverage,",
            "Punchlines drop heavy — dashboards get the message."
        ]
        if any(i["type"] == "anomaly" for i in ins): bars.append("Spike in the data, z-score doing damage,")
        if any(i["type"] == "trend" and i.get("value", 0) > 0 for i in ins): bars.append("Slope aiming sky-high, goals we about to manage.")
        return "\n".join(bars)

    with left:
        st.markdown("**Haiku**")
        st.text_area("Output", _haiku(insights), height=90, key="haiku_out")
        st.markdown("**Rap Verse**")
        st.text_area("Output ", _rap(insights), height=120, key="rap_out")

    with right:
        jokes_for_set = []
        for i in insights:
            seed_val = hash(f"{i['type']}::{i['summary']}") + st.session_state["regen_nonce"]
            random.seed(seed_val)
            j = make_jokes([i], persona, max(1, min(10, base_tone)))[0]
            jokes_for_set.append(_crisis_wrap(j, mood["label"]) + INDUSTRY_TAILS.get(industry, ""))
        def _standup_set(jokes: List[str]) -> str:
            if not jokes: return "Give it up for… data! (pause for nervous Excel laughter)."
            opener = "Hey folks, I just analyzed a dataset — and wow, it has more drama than my inbox."
            closer = "You’ve all been amazing, tip your data scientist, and may your charts always trend up!"
            body = "\n".join([f"- {j}" for j in jokes[:8]])
            return "\n".join([opener, "", body, "", closer])
        st.markdown("**Stand-up Routine (5-min set)**")
        st.text_area("Script", _standup_set(jokes_for_set), height=240, key="set_out")

# ---- Tab: Quiz ----
with tab_quiz:
    st.markdown("### Data Quiz")
    st.caption("Turn your highlights into trivia for meetings.")
    if not insights:
        st.info("No insights to quiz on yet.")
    else:
        total = min(6, len(insights))
        for i, ins in enumerate(insights[:total], start=1):
            q, opts, correct = _quiz_from_insight(ins)
            sel = st.radio(q, opts, key=f"quiz_{i}")
            is_right = (sel == correct)
            st.session_state["quiz_answers"][i] = is_right
            st.caption("✅ Correct!" if is_right else "❌ Not quite — peek at Highlights above.")
        score = sum(1 for v in st.session_state["quiz_answers"].values() if v) if st.session_state["quiz_answers"] else 0
        st.markdown(f"**Score:** {score} / {total}")

# ---- Tab 3: Forecast ----
with tab3:
    st.markdown("### Predictive Story Generation")
    if not metrics:
        st.info("Select at least one numeric metric in the sidebar to enable forecasting.")
    else:
        target_metric = st.selectbox("Metric to forecast", metrics, index=0)
        horizon = st.slider("Horizon (days)", 7, 60, 14, 1)
        if target_metric:
            hist, fut = _linear_forecast(work, target_metric, date_col, horizon=horizon)
            if hist and fut:
                x, y, fit_y = hist
                future_dates, future_vals, a, b = fut
                figf = go.Figure()
                hx = work[date_col].iloc[-len(y):] if (date_col and date_col in work.columns) else list(range(len(y)))
                figf.add_scatter(x=hx, y=y, mode="lines+markers", name="History")
                figf.add_scatter(x=hx, y=fit_y, mode="lines", name="Fit", line=dict(dash="dot"))
                figf.add_scatter(x=future_dates, y=future_vals, mode="lines+markers", name="Forecast")
                figf.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(figf, use_container_width=True)
                st.info(_forecast_story(target_metric, a, future_dates, future_vals))
                st.markdown("#### Scenario Planning")
                sc = _scenario_stories(target_metric, a, pct=0.3)
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"**Best case** slope: {sc['best'][0]:.4f}")
                    st.caption(sc["best"][1])
                    random.seed(hash("bestcase")+st.session_state["regen_nonce"])
                    jk = make_jokes([{"type":"trend","metric":target_metric,"value":abs(a*100),"direction":"increased","summary":"Best-case momentum"}],
                                    persona=persona, tone=min(10, int(base_tone+2)))[0]
                    st.write("🎉 " + _crisis_wrap(jk, mood["label"]))
                with c2:
                    st.error(f"**Worst case** slope: {sc['worst'][0]:.4f}")
                    st.caption(sc["worst"][1])
                    random.seed(hash("worstcase")+st.session_state["regen_nonce"])
                    jk = make_jokes([{"type":"trend","metric":target_metric,"value":abs(a*100),"direction":"decreased","summary":"Worst-case slowdown"}],
                                    persona="Executive Coach", tone=max(1, int(base_tone-2)))[0]
                    st.write("🧯 " + _crisis_wrap(jk, "pessimistic"))
            else:
                st.warning("Not enough data to forecast that metric.")

# ---- Tab 4: Export & Share ----
with tab4:
    st.markdown("### Exports")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate PDF"):
            jokes_for_pdf = []
            for i in insights:
                key_i = f"{i['type']}::{i['summary']}"
                tone_i = max(1, min(10, base_tone + st.session_state['tone_offsets'].get(key_i, 0)))
                jo = make_jokes([i], persona, tone_i)[0]
                jo = _crisis_wrap(jo, mood["label"]) + INDUSTRY_TAILS.get(industry, "")
                jokes_for_pdf.append(jo)
            path = generate_pdf_report(insights, jokes_for_pdf, work.shape, figs_for_pdf if 'figs_for_pdf' in locals() else None)
            if path:
                with open(path, "rb") as f:
                    st.download_button("Download report", f, file_name=os.path.basename(path), mime="application/pdf")
                st.success("Report ready!")
            else:
                st.error("PDF generation failed.")
    with c2:
        st.markdown("**Jokes only (copy)**")
        jokes_only_lines = []
        for i in insights:
            key_i = f"{i['type']}::{i['summary']}"
            tone_i = max(1, min(10, base_tone + st.session_state['tone_offsets'].get(key_i, 0)))
            jo = make_jokes([i], persona, tone_i)[0]
            jo = _crisis_wrap(jo, mood["label"]) + INDUSTRY_TAILS.get(industry, "")
            jokes_only_lines.append("• " + jo)
        st.text_area("", "\n\n".join(jokes_only_lines), height=160)

    st.markdown("### Social Templates")
    best_joke = jokes_only_lines[0] if jokes_only_lines else "My dataset has jokes; my meetings have punchlines."
    def _social_templates(title: str, best_joke: str) -> Dict[str, str]:
        return {
            "twitter": f"{title}: {best_joke} #Data #Analytics #Comedy",
            "linkedin": f"{title}\n\n{best_joke}\n\nTurning insights into memorable stories 🎤📊",
        }
    templates = _social_templates("Weekly Data Roast", best_joke)
    cc1, cc2 = st.columns(2)
    with cc1: st.text_area("Twitter / X", templates["twitter"], height=100)
    with cc2: st.text_area("LinkedIn", templates["linkedin"], height=120)

    st.markdown("### Feedback Export")
    if st.button("Export joke feedback (JSON)"):
        with open("joke_feedback.json","w",encoding="utf-8") as f:
            json.dump(st.session_state["joke_fb"], f, ensure_ascii=False, indent=2)
        with open("joke_feedback.json","rb") as f:
            st.download_button("Download joke_feedback.json", f, file_name="joke_feedback.json", mime="application/json")
        st.success("Saved joke_feedback.json")

# ---- Tab 5: Settings ----
with tab5:
    st.markdown("### Display")
    theme = st.radio("Theme mood", ["Auto (by data)", "Optimistic", "Mixed", "Pessimistic"], horizontal=True)
    if theme != "Auto (by data)":
        override = {"Optimistic": "optimistic", "Mixed": "mixed", "Pessimistic": "pessimistic"}[theme]
        _apply_dynamic_theme({"label": override, "color": "#10b981", "score": mood["score"]})

    st.markdown("### Integrations")
    st.session_state["slack_url"] = st.text_input("Slack Incoming Webhook URL", value=st.session_state["slack_url"], type="password",
                                                  help="Create an Incoming Webhook in Slack and paste the URL here.")

    st.markdown("### About")
    st.markdown(
        "This app turns **insights** into **punchlines**. It includes emotion-aware humor, scenario planning, industry flavoring, "
        "a quiz mode, voice performance, Slack alerts, feedback tracking, and achievements."
    )
    st.markdown('<div class="footer-note">Built with Streamlit, Plotly, pandas, and a pinch of comedic timing.</div>', unsafe_allow_html=True)
