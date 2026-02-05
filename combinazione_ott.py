import streamlit as st
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import plotly.graph_objects as go
import re
from scipy.optimize import minimize

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Asset Allocation: Executive Pro", layout="wide")

# --- STYLING CSS (EXECUTIVE STYLE) ---
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #31333F; }
    [data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    
    /* Metriche In-Sample vs Out-of-Sample */
    .metric-box {
        padding: 15px; border-radius: 8px; border: 1px solid #E0E0E0; margin-bottom: 10px; text-align: center;
    }
    .metric-title { font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: 800; color: #333; }
    .metric-sub { font-size: 12px; color: #888; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #FFFFFF; border-bottom: 1px solid #E0E0E0; }
    .stTabs [aria-selected="true"] { border-top: 3px solid #FF4B4B; color: #000 !important; background-color: #F0F2F6; }
</style>
""", unsafe_allow_html=True)

# --- MOTORE MATEMATICO ---

def load_data(file):
    """
    Caricamento robusto con gestione separatore migliaia.
    """
    try:
        # FIX: thousands='.' è fondamentale per numeri tipo 1.716,94
        df = pd.read_csv(file, sep=';', decimal=',', thousands='.', index_col=0, parse_dates=True, dayfirst=True)
        df.columns = df.columns.str.strip()
        # Conversione forzata a numerico
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Errore tecnico nel caricamento: {e}")
        return None

def detect_frequency(df):
    """Rileva automaticamente la frequenza dei dati per annualizzare correttamente."""
    if len(df) < 2: return 52
    days = (df.index[1] - df.index[0]).days
    if days <= 4: return 252 # Giornaliero
    if days <= 10: return 52 # Settimanale
    if days <= 35: return 12 # Mensile
    return 52 # Default

def get_advanced_stats(weights, returns, freq, rf_rate=0.0):
    """Calcola metriche considerando il Risk Free Rate."""
    weights = np.array(weights)
    port_series = returns.dot(weights)
    
    mean_ret = port_series.mean() * freq
    volatility = port_series.std() * np.sqrt(freq)
    
    # Sharpe con Risk Free
    sharpe = (mean_ret - rf_rate) / volatility if volatility != 0 else 0
    
    negative_returns = port_series[port_series < 0]
    downside_std = negative_returns.std() * np.sqrt(freq)
    sortino = (mean_ret - rf_rate) / downside_std if downside_std != 0 else 0
    
    cumulative = (1 + port_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return mean_ret, volatility, sharpe, sortino, max_drawdown

def get_avg_correlation(data, assets):
    if len(assets) < 2: return 1.0
    corr_matrix = data[list(assets)].corr()
    values = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    return values.mean()

def optimize_portfolio(returns, freq, rf_rate=0.0):
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * freq
    avg_returns = returns.mean() * freq
    
    def objective(weights):
        w = np.array(weights)
        port_ret = np.sum(avg_returns * w)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        # Minimizziamo il Negative Sharpe
        s = (port_ret - rf_rate) / port_vol if port_vol > 0 else 0
        return -s

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    init_guess = [1./n_assets] * n_assets
    
    try:
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except:
        return init_guess

@st.cache_data(show_spinner=False)
def run_optimization_split(data, train_size, k, max_corr, rf_rate):
    """
    Ottimizza sul Train Set, Test sul Test Set.
    Evita il Look-Ahead Bias.
    """
    split_idx = int(len(data) * train_size)
    train_data = data.iloc[:split_idx]
    
    # Calcolo frequenza dinamica
    freq = detect_frequency(data)
    
    assets = data.columns.tolist()
    if len(assets) < k: return None, None, None, None
    
    best_sharpe_train = -np.inf
    best_combo = None
    best_weights = None
    
    # Brute force sulle combinazioni (limitato a k piccoli per performance)
    for combo in itertools.combinations(assets, k):
        # Filtro Correlazione (calcolato sul Train)
        current_corr = get_avg_correlation(train_data, combo)
        
        if current_corr <= max_corr:
            subset_train = train_data[list(combo)].pct_change().dropna()
            
            # Ottimizzazione (SOLO su Train)
            weights = optimize_portfolio(subset_train, freq, rf_rate)
            
            # Valutazione (SOLO su Train per la selezione)
            r, v, s, sort, mdd = get_advanced_stats(weights, subset_train, freq, rf_rate)
            
            if s > best_sharpe_train:
                best_sharpe_train = s
                best_combo = combo
                best_weights = weights
                
    if best_combo:
        # Ora calcoliamo le statistiche finali sia per Train che per Test
        full_subset = data[list(best_combo)].pct_change().dropna()
        
        # Split dei rendimenti
        split_point = int(len(full_subset) * train_size)
        ret_train = full_subset.iloc[:split_point]
        ret_test = full_subset.iloc[split_point:]
        
        stats_train = get_advanced_stats(best_weights, ret_train, freq, rf_rate)
        
        # Se il test set è troppo piccolo, gestiamo l'errore
        if len(ret_test) > 1:
            stats_test = get_advanced_stats(best_weights, ret_test, freq, rf_rate)
        else:
            stats_test = (0,0,0,0,0)
            
        return best_combo, best_weights, stats_train, stats_test
        
    return None, None, None, None

def clean_asset_name(name):
    return re.sub(r'\s*\(.*\)', '', name).strip()

# --- UI APPLICAZIONE ---

st.title("🛡️ Quant Allocation: Reality Check Edition")

with st.sidebar:
    st.header("1. Input Dati")
    uploaded_file = st.file_uploader("Carica CSV (basketai.csv)", type=["csv"])
    
    st.divider()
    st.header("2. Parametri Finanziari")
    rf_input = st.number_input("Risk Free Rate (%)", value=3.0, step=0.1) / 100
    train_split = st.slider("Train/Test Split (Backtest Onesto)", 0.5, 0.9, 0.7, help="Ottimizza sul primo X% dei dati, verifica sul restante.")
    
    st.divider()
    st.header("3. Filtri")
    max_corr_input = st.slider("Max Correlazione Ammessa", 0.0, 1.0, 0.85, step=0.05)

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        freq = detect_frequency(df)
        assets = df.columns.tolist()
        
        # Info Split
        split_idx = int(len(df) * train_split)
        split_date = df.index[split_idx].strftime('%d/%m/%Y')
        
        st.info(f"📊 Frequenza rilevata: {freq} periodi/anno. Split simulazione: {split_date}")
        
        with st.spinner('Esecuzione Ottimizzazione Walk-Forward...'):
            # Manual
            default_asset = assets[0]
            manual_asset = st.selectbox("Benchmark / Asset Manuale", assets)
            
            # Calcoli Manuale
            man_ret = df[[manual_asset]].pct_change().dropna()
            split_p = int(len(man_ret) * train_split)
            man_train_stats = get_advanced_stats([1], man_ret.iloc[:split_p], freq, rf_input)
            man_test_stats = get_advanced_stats([1], man_ret.iloc[split_p:], freq, rf_input)

            # Calcoli Ottimizzati
            l2_combo, l2_w, l2_train, l2_test = run_optimization_split(df, train_split, 2, max_corr_input, rf_input)
            l3_combo, l3_w, l3_train, l3_test = run_optimization_split(df, train_split, 3, max_corr_input, rf_input)

        # --- VISUALIZZAZIONE ---
        
        tab1, tab2, tab3 = st.tabs(["📈 DASHBOARD", "📊 CORRELAZIONI", "📝 DETTAGLI"])
        
        with tab1:
            st.subheader("Performance Reale vs Teorica")
            st.caption("Nota: 'In-Sample' è il passato ottimizzato. 'Out-of-Sample' è la prova del nove (dati non visti dall'algoritmo).")
            
            # Helper per le card
            def draw_card(title, color, train_stats, test_stats, components=None):
                r_tr, v_tr, s_tr, _, _ = train_stats
                r_te, v_te, s_te, _, mdd_te = test_stats
                
                # Composizione stringa
                comp_html = ""
                if components:
                    comp_html = "<div style='font-size:11px; color:#555; margin-bottom:10px;'>" + " + ".join([f"{clean_asset_name(k)} <b>{v*100:.0f}%</b>" for k,v in components]) + "</div>"

                html = f"""
                <div style='background-color:#FFF; padding:15px; border-radius:10px; border-left: 5px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;'>
                    <h4 style='margin:0; font-size:16px;'>{title}</h4>
                    {comp_html}
                    <div style='display:flex; justify-content:space-between; margin-top:15px;'>
                        <div style='text-align:center; width:48%; border-right:1px solid #eee;'>
                            <div class='metric-title'>TRAIN (Ottimizzato)</div>
                            <div class='metric-value' style='color:#888'>{s_tr:.2f}</div>
                            <div class='metric-sub'>SR Ann.</div>
                        </div>
                        <div style='text-align:center; width:48%;'>
                            <div class='metric-title'>TEST (Realtà)</div>
                            <div class='metric-value' style='color:{color}'>{s_te:.2f}</div>
                            <div class='metric-sub'>SR Ann.</div>
                        </div>
                    </div>
                    <div style='margin-top:10px; font-size:13px; text-align:center; color:#333;'>
                        Test Return: <b>{r_te*100:.1f}%</b> | Test DD: <b style='color:#D32F2F'>{mdd_te*100:.1f}%</b>
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1: draw_card("LINEA 1 (Benchmark)", "#777", man_train_stats, man_test_stats)
            with c2: 
                if l2_combo: 
                    comps = list(zip(l2_combo, l2_w))
                    draw_card("LINEA 2 (Best Pair)", "#1C83E1", l2_train, l2_test, comps)
                else: st.warning("Nessuna coppia trovata.")
            with c3:
                if l3_combo:
                    comps = list(zip(l3_combo, l3_w))
                    draw_card("LINEA 3 (Best Triplet)", "#00C853", l3_train, l3_test, comps)
                else: st.warning("Nessuna tripla trovata.")

            st.divider()
            
            # CHART EQUITY LINE
            st.subheader("Simulazione Storica (Equity Line)")
            
            # Ricostruzione serie temporali
            common_idx = df.index
            chart_df = pd.DataFrame(index=common_idx)
            
            # Benchmark
            bench_ret = df[manual_asset].pct_change().fillna(0)
            chart_df["Benchmark"] = (1 + bench_ret).cumprod() * 100
            
            if l2_combo:
                l2_ret = df[list(l2_combo)].pct_change().fillna(0).dot(l2_w)
                chart_df["Best Pair"] = (1 + l2_ret).cumprod() * 100
                
            if l3_combo:
                l3_ret = df[list(l3_combo)].pct_change().fillna(0).dot(l3_w)
                chart_df["Best Triplet"] = (1 + l3_ret).cumprod() * 100

            fig = px.line(chart_df, template='plotly_white')
            
            # Aggiunta linea verticale Split
            split_val = df.index[split_idx]
            fig.add_vline(x=split_val, line_width=2, line_dash="dash", line_color="red")
            fig.add_annotation(x=split_val, y=100, text="INIZIO TEST (FUTURO IGNOTO)", showarrow=True, arrowhead=1)
            
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Matrice di Correlazione (Intero Periodo)")
            sel_assets = [manual_asset]
            if l2_combo: sel_assets.extend(list(l2_combo))
            if l3_combo: sel_assets.extend(list(l3_combo))
            sel_assets = list(set(sel_assets))
            
            corr = df[sel_assets].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr)

else:
    st.info("Attesa caricamento file...")
