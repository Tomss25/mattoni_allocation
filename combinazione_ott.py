import streamlit as st
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import re
from scipy.optimize import minimize

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Asset Allocation: Light Executive", layout="wide")

# --- STYLING CSS AVANZATO (LIGHT MODE - EXECUTIVE STYLE) ---
st.markdown("""
<style>
    /* Sfondo Principale - Bianco Pulito */
    .stApp {
        background-color: #FFFFFF;
        color: #31333F;
    }
    
    /* Sidebar - Grigio Tenue Professionale */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Testi e Header - Nero/Grigio Scuro per massimo contrasto */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    p, div, label, li {
        color: #31333F;
    }
    
    /* --- CUSTOMIZZAZIONE SELECTBOX (Sidebar) --- */
    .stSelectbox label p {
        color: #000000 !important; /* Label nera */
        font-weight: bold;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
    }
    
    /* Tabelle (DataFrame) - Stile Excel Pulito */
    .stDataFrame {
        border: 1px solid #E0E0E0;
    }
    [data-testid="stDataFrameResizable"] {
        background-color: #FFFFFF;
    }
    
    /* Tabs - Stile Moderno Chiaro */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #FFFFFF;
        border-bottom: 1px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #666666;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0F2F6 !important;
        color: #000000 !important;
        border-top: 3px solid #FF4B4B; /* Highlight Rosso Streamlit o Blu Corporate */
        border-bottom: 1px solid #F0F2F6;
    }

    /* Divisori */
    hr {
        border-color: #E0E0E0;
    }
    
    /* Messaggi di Alert */
    .stAlert {
        background-color: #F0F2F6;
        color: #31333F;
        border: 1px solid #D1D1D1;
    }
</style>
""", unsafe_allow_html=True)

# --- MOTORE MATEMATICO (INVARIATO) ---

def load_data(file):
    """Caricamento robusto (Sep=; Dec=, Date=GG/MM/AAAA)."""
    try:
        df = pd.read_csv(file, sep=';', decimal=',', index_col=0, parse_dates=True, dayfirst=True)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        return None

def clean_asset_name(name):
    """Rimuove il rumore dal nome dell'asset."""
    clean = re.sub(r'\s*\(.*\)', '', name)
    return clean.strip()

def get_advanced_stats(weights, returns):
    """Calcola metriche avanzate: Rendimento, Volatilità, Sharpe, Sortino, MDD."""
    weights = np.array(weights)
    port_series = returns.dot(weights)
    
    annual_factor = 52
    mean_ret = port_series.mean() * annual_factor
    volatility = port_series.std() * np.sqrt(annual_factor)
    
    sharpe = mean_ret / volatility if volatility != 0 else 0
    
    negative_returns = port_series[port_series < 0]
    downside_std = negative_returns.std() * np.sqrt(annual_factor)
    sortino = mean_ret / downside_std if downside_std != 0 else 0
    
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

def optimize_portfolio(returns):
    n_assets = len(returns.columns)
    def objective(weights):
        w = np.array(weights)
        ret = np.sum(returns.mean() * w) * 52
        vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 52, w)))
        s = ret / vol if vol > 0 else 0
        return -s

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = [1./n_assets for _ in range(n_assets)]
    
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

@st.cache_data(show_spinner=False)
def find_best_optimized_combination(data, k, max_corr_threshold=1.0):
    assets = data.columns.tolist()
    if len(assets) < k: return None, None, (0,0,0,0,0)
    
    best_sharpe = -np.inf
    best_combo = None
    best_weights = None
    best_full_stats = None
    
    for combo in itertools.combinations(assets, k):
        # Filtro correlazione
        current_corr = get_avg_correlation(data, combo)
        
        if current_corr <= max_corr_threshold:
            subset = data[list(combo)].pct_change().dropna()
            weights = optimize_portfolio(subset)
            r, v, s, sort, mdd = get_advanced_stats(weights, subset)
            
            if s > best_sharpe:
                best_sharpe = s
                best_combo = combo
                best_weights = weights
                best_full_stats = (r, v, s, sort, mdd)
            
    return best_combo, best_weights, best_full_stats

def format_composition(assets, weights):
    items = []
    sorted_pairs = sorted(zip(assets, weights), key=lambda x: x[1], reverse=True)
    for a, w in sorted_pairs:
        if w > 0.001: 
            clean_name = clean_asset_name(a)
            items.append(f"{clean_name} ({w*100:.0f}%)")
    return " + ".join(items)

# --- UI APPLICAZIONE ---

st.title("🛡️ Asset Optimizer: Executive Dashboard")

# SIDEBAR
with st.sidebar:
    st.header("1. Data Feed")
    uploaded_file = st.file_uploader("Carica CSV (basketai.csv)", type=["csv"])
    manual_placeholder = st.empty()
    
    st.divider()
    st.header("3. Filtri Strategici")
    st.markdown("Definisci il compromesso accettabile:")
    max_corr_input = st.slider(
        "Max Correlazione Ammessa", 
        min_value=0.0, 
        max_value=1.0, 
        value=1.0, 
        step=0.05
    )

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        assets = df.columns.tolist()
        
        with st.spinner('Calcolo Ottimizzazione e Analisi Metodologica...'):
            # 1. Best Single Asset
            temp_sharpes = {}
            for a in assets:
                r_t = df[[a]].pct_change().dropna()
                _, _, s_t, _, _ = get_advanced_stats([1], r_t)
                temp_sharpes[a] = s_t
            
            best_single = max(temp_sharpes, key=temp_sharpes.get)
            
            # UI Manuale
            default_idx = assets.index(best_single)
            manual_asset = manual_placeholder.selectbox("2. Linea 1 (Manuale)", assets, index=default_idx)
            
            # Dati Linea 1
            l1_ret_frame = df[[manual_asset]].pct_change().dropna()
            l1_stats = get_advanced_stats([1], l1_ret_frame)
            l1_corr = 1.0
            
            # 2. Best Pair Optimized
            pair_assets, pair_weights, pair_stats = find_best_optimized_combination(df, 2, max_corr_input)
            if pair_assets:
                l2_corr = get_avg_correlation(df, pair_assets)
                l2_series = df[list(pair_assets)].pct_change().dropna().dot(pair_weights)
            
            # 3. Best Triplet Optimized
            triplet_assets, triplet_weights, triplet_stats = find_best_optimized_combination(df, 3, max_corr_input)
            if triplet_assets:
                l3_corr = get_avg_correlation(df, triplet_assets)
                l3_series = df[list(triplet_assets)].pct_change().dropna().dot(triplet_weights)

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ DASHBOARD", "2️⃣ CORRELAZIONI", "3️⃣ BACKTEST", "📘 METODOLOGIA"])

        # --- TAB 1: DASHBOARD ---
        with tab1:
            st.subheader("Allocazione Ottimale (Vincolata)")
            if max_corr_input < 1.0:
                st.info(f"💡 Filtro Attivo: Combinazioni limitate a correlazione < {max_corr_input}.")
            
            table_data = []
            def make_row(label, asset_list, weights, corr, stats):
                r, v, s, sort, mdd = stats
                if isinstance(asset_list, str): comp_str = f"{clean_asset_name(asset_list)} (100%)"
                else: comp_str = format_composition(asset_list, weights)
                return {
                    "Strategia": label,
                    "Allocazione (Pesi Ottimali)": comp_str,
                    "Corr. Media": f"{corr:.2f}" if isinstance(corr, float) else "N/A",
                    "Rend. Annuo": f"{r*100:.1f}%",
                    "Max DD": f"{mdd*100:.1f}%",
                    "Sharpe": f"{s:.2f}",
                    "Sortino": f"{sort:.2f}"
                }
            
            table_data.append(make_row("LINEA 1 (Manuale)", manual_asset, [1], l1_corr, l1_stats))
            if pair_assets: table_data.append(make_row("LINEA 2 (Best Pair)", pair_assets, pair_weights, l2_corr, pair_stats))
            else: st.warning("Nessuna coppia trovata con i filtri attuali.")
            if triplet_assets: table_data.append(make_row("LINEA 3 (Best Triplet)", triplet_assets, triplet_weights, l3_corr, triplet_stats))
            
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
            
            st.divider()
            st.markdown("### 📊 Performance vs Rischio")
            col1, col2, col3 = st.columns(3)
            
            # STILE CSS AGGIORNATO PER LIGHT MODE (Carte con ombra leggera)
            box_style = """
            <div style='background-color: #FFFFFF; padding: 20px; border-radius: 10px; border: 1px solid #E0E0E0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;'>
                <h4 style='color: #666666; margin:0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;'>{title}</h4>
                <div style='margin: 15px 0;'>
                    <span style='font-size: 32px; font-weight: 800; color: {color};'>SR {sharpe}</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 10px 0; border-top: 1px solid #F0F0F0; border-bottom: 1px solid #F0F0F0; font-size: 14px; color: #333333;'>
                    <span>Rendimento: <b>{ret}</b></span>
                    <span>Max DD: <b style='color: #D32F2F;'>{mdd}</b></span>
                </div>
                <div style='margin-top: 10px; font-size: 12px; color: #888888;'>Sortino Ratio: <b>{sort}</b></div>
            </div>
            """
            def render_box(col, title, color, stats):
                r, v, s, sort, mdd = stats
                col.markdown(box_style.format(title=title, color=color, sharpe=f"{s:.2f}", ret=f"{r*100:.1f}%", mdd=f"{mdd*100:.1f}%", sort=f"{sort:.2f}"), unsafe_allow_html=True)

            render_box(col1, "LINEA 1", "#FF4B4B", l1_stats)
            if pair_assets: render_box(col2, "LINEA 2", "#1C83E1", pair_stats)
            if triplet_assets: render_box(col3, "LINEA 3", "#00C853", triplet_stats)

        # --- TAB 2: CORRELAZIONI ---
        with tab2:
            st.subheader("1. Asset Selezionati")
            unique_assets = list(set([manual_asset] + list(pair_assets or []) + list(triplet_assets or [])))
            clean_labels = {a: clean_asset_name(a) for a in unique_assets}
            # Template Plotly White
            fig_corr = px.imshow(df[unique_assets].rename(columns=clean_labels).corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template='plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("---")
            st.subheader("2. Intero Paniere")
            all_clean = {a: clean_asset_name(a) for a in assets}
            # Template Plotly White
            fig_full = px.imshow(df.rename(columns=all_clean).corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template='plotly_white')
            fig_full.update_layout(height=600 if len(assets) < 15 else 900)
            st.plotly_chart(fig_full, use_container_width=True)

        # --- TAB 3: BACKTEST ---
        with tab3:
            st.subheader("Simulazione Storica (Equity Line)")
            common_idx = l1_ret_frame.index
            if pair_assets: common_idx = common_idx.intersection(l2_series.index)
            if triplet_assets: common_idx = common_idx.intersection(l3_series.index)
            
            chart_df = pd.DataFrame(index=common_idx)
            chart_df[f"L1: {clean_asset_name(manual_asset)}"] = (1 + l1_ret_frame.loc[common_idx][manual_asset]).cumprod() * 100
            if pair_assets: chart_df["L2: Best Pair"] = (1 + l2_series.loc[common_idx]).cumprod() * 100
            if triplet_assets: chart_df["L3: Best Triplet"] = (1 + l3_series.loc[common_idx]).cumprod() * 100
            
            # Template Plotly White + Legenda NERA (Default)
            fig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, template='plotly_white')
            fig.update_layout(
                xaxis_title=None, 
                yaxis_title="Valore (Base 100)", 
                legend=dict(
                    orientation="h", 
                    y=1.1, 
                    title=None
                    # Non forziamo più il colore bianco, plotly_white usa il nero di default
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 4: METODOLOGIA ---
        with tab4:
            st.subheader("📘 Logica di Funzionamento del Modello")
            
            st.markdown("""
            ### 1. Come vengono scelti gli asset?
            Il programma non "sceglie" in base a simpatie o trend. Usa un approccio puramente matematico basato sulla **Modern Portfolio Theory (MPT)**.
            
            * **Obiettivo Primario:** Massimizzare lo **Sharpe Ratio**.
            * **Cosa significa:** Il software cerca la combinazione di asset e pesi che ha storicamente generato il **massimo rendimento per ogni unità di rischio** assunta. Non cerca il rendimento massimo assoluto (che spesso nasconde rischi folli), ma l'efficienza.
            
            ### 2. Come vengono calcolati i pesi? (Algoritmo SLSQP)
            Utilizziamo un ottimizzatore non lineare (`Sequential Least Squares Programming`) che testa migliaia di combinazioni di percentuali per rispondere a questa domanda:
            > *"Qual è la dose esatta di Asset A e Asset B che, miscelata insieme, rende la curva dei rendimenti più stabile possibile e inclinata verso l'alto?"*
            
            ### 3. I Filtri Strategici
            Se hai attivato il filtro **"Max Correlazione"** nella barra laterale, il modello applica una censura preventiva:
            1.  Analizza tutte le possibili coppie/terne.
            2.  **SCARTA** immediatamente quelle che si muovono troppo insieme (correlazione > soglia).
            3.  Solo tra le sopravvissute (quelle diversificate), cerca la più efficiente.
            
            ---
            
            ### 4. Glossario Brutale (Per non mentire a se stessi)
            
            | Metrica | Cosa ti dice (Traduzione Onesta) |
            | :--- | :--- |
            | **Sharpe Ratio** | Il voto in pagella del portafoglio. Sopra 1.0 è buono, sopra 2.0 è eccellente. Sotto 0.5 stai rischiando per nulla. |
            | **Max Drawdown** | **Il dolore.** La percentuale massima che avresti perso dai massimi se avessi comprato nel momento peggiore e venduto nel momento peggiore. Se vedi -30% e non sei disposto a perdere 1/3 del capitale, lascia stare. |
            | **Sortino Ratio** | Come lo Sharpe, ma ignora la volatilità "buona" (quando il titolo sale). È un giudice più severo e preciso per gli investitori avversi alle perdite. |
            | **Correlazione** | **0.0 - 0.5:** Diversificazione reale (sicurezza). <br> **0.7 - 1.0:** Falsa diversificazione (se cade uno, cade anche l'altro). |
            
            ⚠️ **DISCLAIMER:** *Questo modello soffre di "Look-Ahead Bias". Ha ottimizzato i pesi guardando i dati del passato. Il futuro non sarà identico. Usa questi risultati come indicazione di potenziale strutturale, non come garanzia di profitto.*
            """)

    else:
        st.error("File non valido.")
else:
    st.info("Carica il file CSV.")