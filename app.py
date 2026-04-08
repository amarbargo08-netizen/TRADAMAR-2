import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TRADAMAR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark Pro TradingView Style
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .stApp { background-color: #0d1117; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: #8b949e !important; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; }

  /* Header Banner */
  .tradamar-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
  }
  .tradamar-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #238636, #1f6feb, #238636);
  }
  .tradamar-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #f0f6fc;
  }
  .tradamar-logo span { color: #238636; }
  .tradamar-sub {
    font-size: 0.78rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .tradamar-badge {
    background: #238636;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
  }

  /* Metric Cards */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }
  .metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #30363d; }
  .metric-card .label {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
  }
  .metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1;
  }
  .metric-card .sub {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 4px;
  }
  .metric-card.green { border-left: 3px solid #238636; }
  .metric-card.red   { border-left: 3px solid #da3633; }
  .metric-card.blue  { border-left: 3px solid #1f6feb; }
  .metric-card.gold  { border-left: 3px solid #e3b341; }
  .metric-card.purple{ border-left: 3px solid #8957e5; }
  .value.green { color: #3fb950; }
  .value.red   { color: #f85149; }
  .value.gold  { color: #e3b341; }

  /* Section titles */
  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 24px 0 12px 0;
    padding-left: 12px;
    border-left: 2px solid #238636;
  }

  /* Signal Table */
  .signal-row {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
  }
  .badge-long  { background: rgba(35,134,54,0.2); color: #3fb950; border:1px solid #238636; padding:3px 10px; border-radius:4px; font-size:0.7rem; }
  .badge-short { background: rgba(218,54,51,0.2); color: #f85149; border:1px solid #da3633; padding:3px 10px; border-radius:4px; font-size:0.7rem; }

  /* Divider */
  hr { border-color: #21262d; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius:8px; border:1px solid #21262d; }
  .stTabs [data-baseweb="tab"] { color:#8b949e; font-family:'JetBrains Mono',monospace; font-size:0.8rem; letter-spacing:0.06em; }
  .stTabs [aria-selected="true"] { color:#f0f6fc !important; }

  /* Buttons */
  .stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    transition: background 0.2s;
    width: 100%;
  }
  .stButton > button:hover { background: #2ea043; border:none; }

  /* Download button */
  .stDownloadButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    width: 100%;
  }
  .stDownloadButton > button:hover { border-color:#8b949e; }

  /* Spinner */
  .stSpinner > div { border-top-color: #238636 !important; }

  /* Info / Warning boxes */
  .stAlert { background: #161b22; border:1px solid #21262d; border-radius:8px; }

  /* No top padding */
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR — PARAMETERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Marché**")
    symbol = st.selectbox("Actif", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AAPL", "TSLA", "EURUSD=X"], index=0)
    interval = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
    period   = st.selectbox("Période", ["3mo", "6mo", "1y", "2y"], index=2)

    st.markdown("---")
    st.markdown("**Détection de structures**")
    window_size    = st.slider("Window pivots", 5, 30, 10, 1)
    min_points     = st.slider("Min points / structure", 2, 5, 3, 1)
    break_tol      = st.slider("Tolérance cassure (%)", 0.1, 2.0, 0.5, 0.1) / 100

    st.markdown("---")
    st.markdown("**Modèle 1 — Breakout**")
    rr_ratio = st.slider("Risk/Reward ratio", 1.0, 5.0, 2.0, 0.5)
    zone_pct  = st.slider("Zone SL (%)", 5, 30, 15, 5) / 100

    st.markdown("---")
    st.markdown("**Modèle 2 — Rebond**")
    zone_validation = st.slider("Zone contact (%)", 10, 30, 15, 5) / 100
    seuil_entree    = st.slider("Seuil entrée (%)", 15, 35, 21, 1) / 100
    seuil_tp        = st.slider("Seuil TP (%)", 50, 95, 78, 1) / 100
    marge_sl        = st.slider("Marge SL ext (%)", 2, 15, 5, 1) / 100

    st.markdown("---")
    st.markdown("**Filtres**")
    one_position    = st.toggle("Filtre 1 position active", value=True)
    model_breakout  = st.toggle("Activer Modèle 1 (Breakout)", value=True)
    model_rebond    = st.toggle("Activer Modèle 2 (Rebond)", value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Lancer l'analyse")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="tradamar-header">
  <div>
    <div class="tradamar-logo">TRAD<span>AMAR</span></div>
    <div class="tradamar-sub">Structure Scanner · Technical Analysis Engine</div>
  </div>
  <div style="text-align:right">
    <div class="tradamar-badge">v2.0 · LIVE</div>
    <div style="font-size:0.72rem;color:#8b949e;margin-top:6px;font-family:'JetBrains Mono',monospace">{symbol} · {interval} · {period}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CORE ENGINE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_data(sym, ivl, per):
    try:
        fetch_interval = "1h" if ivl == "4h" else ivl
        ticker = yf.Ticker(sym)
        df = ticker.history(period=per, interval=fetch_interval)

        if df is None or df.empty:
            df = yf.download(tickers=sym, period=per, interval=fetch_interval, progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.capitalize() for c in df.columns]
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[needed].dropna()

        if ivl == "4h" and not df.empty:
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            if "Volume" in df.columns:
                agg["Volume"] = "sum"
            df = df.resample("4h").agg(agg).dropna()

        return df if not df.empty else None

    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        return None


def detect_pivots(df, window):
    high_idx = argrelextrema(df["High"].values, np.greater_equal, order=window)[0]
    low_idx  = argrelextrema(df["Low"].values,  np.less_equal,   order=window)[0]
    return high_idx, low_idx


def detect_structures(df, high_idx, low_idx, min_pts, break_tol):
    structures = []
    h_list = list(high_idx)
    l_list = list(low_idx)
    last_break_idx = 0
    i = 0

    while i < len(h_list) - min_pts:
        h_group = [idx for idx in h_list if idx >= last_break_idx][i:i + min_pts]
        if len(h_group) < min_pts:
            break

        l_group = [idx for idx in l_list if idx >= h_group[0] and idx <= h_group[-1] + 25][:min_pts]
        if len(l_group) < min_pts:
            i += 1
            continue

        s_h, int_h, _, _, _ = linregress(h_group, df["High"].values[h_group])
        s_l, int_l, _, _, _ = linregress(l_group, df["Low"].values[l_group])

        start_s = min(h_group[0], l_group[0])
        end_s   = max(h_group[-1], l_group[-1])

        gap_init  = (s_h * start_s + int_h) - (s_l * start_s + int_l)
        gap_final = (s_h * end_s   + int_h) - (s_l * end_s   + int_l)

        if gap_init > 0 and gap_final > 0 and gap_final <= gap_init * 1.1:
            j = end_s
            while j < len(df) - 1:
                price   = float(df["Close"].values[j])
                limit_h = s_h * j + int_h
                limit_l = s_l * j + int_l
                if price > limit_h * (1 + break_tol) or price < limit_l * (1 - break_tol):
                    break
                j += 1

            if j - start_s >= min_pts * 3:
                structures.append({
                    "start": start_s, "end": j,
                    "s_h": s_h, "int_h": int_h,
                    "s_l": s_l, "int_l": int_l,
                })
                last_break_idx = j

        i += 1

    return structures


def generate_signals_breakout(df, structures, zone_pct, rr, one_pos):
    signals = []
    in_position = False

    for s in structures:
        idx_break = s["end"]
        if idx_break >= len(df):
            continue
        if one_pos and in_position:
            continue

        price_entry = float(df["Close"].values[idx_break])
        res_t = s["s_h"] * idx_break + s["int_h"]
        sup_t = s["s_l"] * idx_break + s["int_l"]
        H_t   = res_t - sup_t

        if H_t <= 0:
            continue

        limit_rose  = res_t - zone_pct * H_t
        limit_verte = sup_t + zone_pct * H_t

        if price_entry > res_t:
            sl   = limit_rose
            risk = price_entry - sl
            tp   = price_entry + risk * rr
            signals.append({
                "date": df.index[idx_break], "modele": "Breakout",
                "type": "LONG", "entry": round(price_entry, 4),
                "sl": round(sl, 4), "tp": round(tp, 4), "rr": rr,
            })
            in_position = True

        elif price_entry < sup_t:
            sl   = limit_verte
            risk = sl - price_entry
            tp   = price_entry - risk * rr
            signals.append({
                "date": df.index[idx_break], "modele": "Breakout",
                "type": "SHORT", "entry": round(price_entry, 4),
                "sl": round(sl, 4), "tp": round(tp, 4), "rr": rr,
            })
            in_position = True

    return signals


def generate_signals_rebond(df, structures, zone_val, seuil_e, seuil_tp, marge_sl, one_pos):
    signals = []
    in_position = False

    for s in structures:
        indices = np.arange(s["start"], s["end"])
        has_rose = has_verte = False

        for idx in indices:
            if idx >= len(df):
                break
            if one_pos and in_position:
                break

            p_high  = float(df["High"].values[idx])
            p_low   = float(df["Low"].values[idx])
            p_close = float(df["Close"].values[idx])

            res_t = s["s_h"] * idx + s["int_h"]
            sup_t = s["s_l"] * idx + s["int_l"]
            H_t   = res_t - sup_t
            if H_t <= 0:
                continue

            lim_rose  = res_t - zone_val * H_t
            lim_verte = sup_t + zone_val * H_t

            if p_high >= lim_rose:  has_rose  = True
            if p_low  <= lim_verte: has_verte = True

            if has_rose and p_close < (res_t - seuil_e * H_t):
                entry = p_close
                sl = res_t + marge_sl * H_t
                tp = res_t - seuil_tp * H_t
                rr = round(abs(entry - tp) / abs(sl - entry), 2) if abs(sl - entry) > 0 else 0
                signals.append({
                    "date": df.index[idx], "modele": "Rebond",
                    "type": "SHORT", "entry": round(entry, 4),
                    "sl": round(sl, 4), "tp": round(tp, 4), "rr": rr,
                })
                has_rose = False
                in_position = True

            elif has_verte and p_close > (sup_t + seuil_e * H_t):
                entry = p_close
                sl = sup_t - marge_sl * H_t
                tp = sup_t + seuil_tp * H_t
                rr = round(abs(entry - tp) / abs(sl - entry), 2) if abs(sl - entry) > 0 else 0
                signals.append({
                    "date": df.index[idx], "modele": "Rebond",
                    "type": "LONG", "entry": round(entry, 4),
                    "sl": round(sl, 4), "tp": round(tp, 4), "rr": rr,
                })
                has_verte = False
                in_position = True

    return signals


def compute_stats(signals, df):
    if not signals:
        return {}

    results = []
    for sig in signals:
        date  = sig["date"]
        entry = sig["entry"]
        sl    = sig["sl"]
        tp    = sig["tp"]
        typ   = sig["type"]

        future = df[df.index > date]
        outcome = "OPEN"
        pnl_pct = 0.0

        for _, row in future.iterrows():
            if typ == "LONG":
                if row["Low"] <= sl:
                    outcome = "LOSS"
                    pnl_pct = (sl - entry) / entry * 100
                    break
                if row["High"] >= tp:
                    outcome = "WIN"
                    pnl_pct = (tp - entry) / entry * 100
                    break
            else:
                if row["High"] >= sl:
                    outcome = "LOSS"
                    pnl_pct = (entry - sl) / entry * 100
                    break
                if row["Low"] <= tp:
                    outcome = "WIN"
                    pnl_pct = (entry - tp) / entry * 100
                    break

        results.append({**sig, "outcome": outcome, "pnl_pct": round(pnl_pct, 2)})

    df_res   = pd.DataFrame(results)
    closed   = df_res[df_res["outcome"].isin(["WIN", "LOSS"])]
    wins     = (closed["outcome"] == "WIN").sum()
    losses   = (closed["outcome"] == "LOSS").sum()
    winrate  = round(wins / len(closed) * 100, 1) if len(closed) > 0 else 0
    avg_rr   = round(df_res["rr"].mean(), 2)
    net_pnl  = round(df_res["pnl_pct"].sum(), 2)

    return {
        "df": df_res,
        "total": len(df_res),
        "wins": wins,
        "losses": losses,
        "open": (df_res["outcome"] == "OPEN").sum(),
        "winrate": winrate,
        "avg_rr": avg_rr,
        "net_pnl": net_pnl,
    }


def build_chart(df, structures, signals_all, high_idx, low_idx):
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name=symbol, showlegend=False
    ))

    # Pivot dots
    pivot_high_y = [df["High"].values[i] if i in set(high_idx) else None for i in range(len(df))]
    pivot_low_y  = [df["Low"].values[i]  if i in set(low_idx)  else None for i in range(len(df))]

    fig.add_trace(go.Scatter(
        x=df.index, y=pivot_high_y, mode="markers",
        marker=dict(color="#1f6feb", size=7, symbol="circle"),
        name="Pivot Haut", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=pivot_low_y, mode="markers",
        marker=dict(color="#1f6feb", size=7, symbol="circle-open", line=dict(width=2)),
        name="Pivot Bas", showlegend=True
    ))

    # Structures
    for s in structures:
        indices = np.arange(s["start"], min(s["end"], len(df)))
        dates   = df.index[indices]
        res = s["s_h"] * indices + s["int_h"]
        sup = s["s_l"] * indices + s["int_l"]
        h_v = res - sup
        z_rose  = res - 0.15 * h_v
        z_verte = sup + 0.15 * h_v

        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([res, z_rose[::-1]]),
            fill="toself", fillcolor="rgba(248,81,73,0.08)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([sup, z_verte[::-1]]),
            fill="toself", fillcolor="rgba(63,185,80,0.06)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(x=dates, y=res, line=dict(color="rgba(248,81,73,0.6)", width=1, dash="dot"), showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=sup, line=dict(color="rgba(63,185,80,0.6)", width=1, dash="dot"), showlegend=False))

    # Signals
    for sig in signals_all:
        color   = "#3fb950" if sig["type"] == "LONG" else "#f85149"
        sym_mk  = "triangle-up" if sig["type"] == "LONG" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[sig["date"]], y=[sig["entry"]],
            mode="markers+text",
            marker=dict(color=color, size=12, symbol=sym_mk, line=dict(width=1, color="#0d1117")),
            text=[sig["type"]], textposition="top center",
            textfont=dict(color=color, size=9, family="JetBrains Mono"),
            name=f"{sig['modele']} {sig['type']}", showlegend=False,
            hovertemplate=f"<b>{sig['modele']} {sig['type']}</b><br>Entrée: {sig['entry']}<br>SL: {sig['sl']}<br>TP: {sig['tp']}<br>RR: {sig['rr']}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            rangeslider_visible=False,
            gridcolor="#21262d", gridwidth=1,
            showspikes=True, spikecolor="#30363d", spikewidth=1,
        ),
        yaxis=dict(
            gridcolor="#21262d", gridwidth=1,
            showspikes=True, spikecolor="#30363d",
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#21262d", borderwidth=1,
            font=dict(family="JetBrains Mono", size=10, color="#8b949e")
        ),
        dragmode="zoom",
        hovermode="x",
    )
    return fig


# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    with st.spinner("Chargement des données et analyse en cours..."):
        df = load_data(symbol, interval, period)
        if df is None or df.empty:
            st.error("❌ Impossible de charger les données.")
        else:
            high_idx, low_idx = detect_pivots(df, window_size)
            structures = detect_structures(df, high_idx, low_idx, min_points, break_tol)

            signals_all = []
            if model_breakout:
                signals_all += generate_signals_breakout(df, structures, zone_pct, rr_ratio, one_position)
            if model_rebond:
                signals_all += generate_signals_rebond(df, structures, zone_validation, seuil_entree, seuil_tp, marge_sl, one_position)

            signals_all.sort(key=lambda x: x["date"])
            stats = compute_stats(signals_all, df)
            fig   = build_chart(df, structures, signals_all, high_idx, low_idx)

            st.session_state.results = {
                "df": df, "structures": structures,
                "signals": signals_all, "stats": stats,
                "fig": fig, "high_idx": high_idx, "low_idx": low_idx,
            }

# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
if st.session_state.results:
    R = st.session_state.results
    stats = R["stats"]

    # ── KPI Cards ──
    winrate_color = "green" if stats.get("winrate", 0) >= 50 else "red"
    pnl_color     = "green" if stats.get("net_pnl", 0) >= 0  else "red"

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card blue">
        <div class="label">Structures</div>
        <div class="value">{len(R['structures'])}</div>
        <div class="sub">détectées</div>
      </div>
      <div class="metric-card gold">
        <div class="label">Signaux</div>
        <div class="value">{stats.get('total', 0)}</div>
        <div class="sub">{stats.get('open', 0)} ouverts</div>
      </div>
      <div class="metric-card {'green' if winrate_color=='green' else 'red'}">
        <div class="label">Winrate</div>
        <div class="value {winrate_color}">{stats.get('winrate', 0)}%</div>
        <div class="sub">{stats.get('wins', 0)}W / {stats.get('losses', 0)}L</div>
      </div>
      <div class="metric-card purple">
        <div class="label">RR Moyen</div>
        <div class="value">{stats.get('avg_rr', 0)}</div>
        <div class="sub">risk/reward</div>
      </div>
      <div class="metric-card {'green' if pnl_color=='green' else 'red'}">
        <div class="label">PnL net</div>
        <div class="value {pnl_color}">{'+' if stats.get('net_pnl',0)>=0 else ''}{stats.get('net_pnl', 0)}%</div>
        <div class="sub">total cumulé</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📈  Graphique", "📋  Signaux", "📊  Statistiques"])

    with tab1:
        st.plotly_chart(R["fig"], use_container_width=True, config={
            "scrollZoom": True, "displayModeBar": True,
            "modeBarButtonsToRemove": ["autoScale2d"],
        })

    with tab2:
        st.markdown('<div class="section-title">Tous les signaux générés</div>', unsafe_allow_html=True)
        df_sig = stats.get("df")

        if df_sig is not None and not df_sig.empty:
            # Filtres rapides
            c1, c2, c3 = st.columns(3)
            with c1:
                f_modele = st.multiselect("Modèle", ["Breakout", "Rebond"], default=["Breakout", "Rebond"])
            with c2:
                f_type = st.multiselect("Direction", ["LONG", "SHORT"], default=["LONG", "SHORT"])
            with c3:
                f_outcome = st.multiselect("Résultat", ["WIN", "LOSS", "OPEN"], default=["WIN", "LOSS", "OPEN"])

            df_filtered = df_sig[
                df_sig["modele"].isin(f_modele) &
                df_sig["type"].isin(f_type) &
                df_sig["outcome"].isin(f_outcome)
            ].copy()

            df_filtered["date"] = pd.to_datetime(df_filtered["date"]).dt.strftime("%Y-%m-%d %H:%M")

            # Couleur outcome
            def style_outcome(val):
                if val == "WIN":  return "color: #3fb950; font-weight:600"
                if val == "LOSS": return "color: #f85149; font-weight:600"
                return "color: #e3b341"

            def style_type(val):
                return "color: #3fb950" if val == "LONG" else "color: #f85149"

            styled = df_filtered.style\
                .applymap(style_outcome, subset=["outcome"])\
                .applymap(style_type, subset=["type"])\
                .format({"entry": "{:.4f}", "sl": "{:.4f}", "tp": "{:.4f}", "rr": "{:.2f}", "pnl_pct": "{:+.2f}%"})

            st.dataframe(styled, use_container_width=True, height=420)

            # Export CSV
            csv_buf = io.BytesIO()
            df_filtered.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️  Exporter CSV",
                data=csv_buf.getvalue(),
                file_name=f"tradamar_signals_{symbol}_{interval}.csv",
                mime="text/csv"
            )
        else:
            st.info("Aucun signal généré avec ces paramètres.")

    with tab3:
        st.markdown('<div class="section-title">Performance globale</div>', unsafe_allow_html=True)
        df_res = stats.get("df")

        if df_res is not None and not df_res.empty:
            # PnL cumulé
            df_closed = df_res[df_res["outcome"].isin(["WIN", "LOSS"])].copy()
            df_closed = df_closed.sort_values("date")
            df_closed["pnl_cum"] = df_closed["pnl_pct"].cumsum()

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=df_closed["date"], y=df_closed["pnl_cum"],
                fill="tozeroy",
                fillcolor="rgba(35,134,54,0.12)" if df_closed["pnl_cum"].iloc[-1] >= 0 else "rgba(218,54,51,0.12)",
                line=dict(color="#238636" if df_closed["pnl_cum"].iloc[-1] >= 0 else "#da3633", width=2),
                name="PnL cumulé (%)"
            ))
            fig_pnl.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=300, margin=dict(l=0,r=0,t=20,b=0),
                xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                title=dict(text="Courbe PnL cumulé (%)", font=dict(family="JetBrains Mono", size=12, color="#8b949e"))
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Par modèle
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-title">Par modèle</div>', unsafe_allow_html=True)
                grp = df_res.groupby("modele").agg(
                    Signaux=("type", "count"),
                    Wins=("outcome", lambda x: (x == "WIN").sum()),
                    Losses=("outcome", lambda x: (x == "LOSS").sum()),
                    PnL_net=("pnl_pct", "sum"),
                    RR_moy=("rr", "mean")
                ).round(2)
                st.dataframe(grp, use_container_width=True)

            with c2:
                st.markdown('<div class="section-title">Par direction</div>', unsafe_allow_html=True)
                grp2 = df_res.groupby("type").agg(
                    Signaux=("entry", "count"),
                    Wins=("outcome", lambda x: (x == "WIN").sum()),
                    PnL_net=("pnl_pct", "sum"),
                ).round(2)
                st.dataframe(grp2, use_container_width=True)

        else:
            st.info("Lancez l'analyse pour voir les statistiques.")

else:
    # Welcome state
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color:#8b949e;">
      <div style="font-size:3rem; margin-bottom:16px">📈</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; color:#8b949e; letter-spacing:0.1em">
        CONFIGURE LES PARAMÈTRES ET LANCE L'ANALYSE
      </div>
      <div style="font-size:0.8rem; margin-top:10px; color:#484f58">
        Structures · Breakouts · Rebonds · Signaux · Stats
      </div>
    </div>
    """, unsafe_allow_html=True)
