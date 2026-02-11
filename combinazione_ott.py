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
    
    /* Testi e Header */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    p, div, label, li {
        color: #31333F;
    }
    
    /* Selectbox e Input */
    .stSelectbox label p {
        color: #000000 !important;
        font-weight: bold;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
    }
    
    /* Tabelle */
    .stDataFrame {
        border: 1px solid #E0E0E0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #FFFFFF;
        border-bottom: 1px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        color: #666666;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0F2F6 !important;
        color: #000000 !important;
        border-top: 3px solid #FF4B4B;
        border-bottom: 1px solid #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)

# --- MOTORE MATEMATICO ---

@st.cache_data(show_spinner=False)
def load_data(file):
    """
    Caricamento Intelligente Universale:
    1. Gestisce CSV (Sep ;/,) ed Excel (.xlsx).
    2. Trova automaticamente la riga di Header (salta righe vuote).
    3. Rileva se il file è Trasposto (Date in colonna) e lo ruota.
    4. Gestisce formati numerici misti (EU/US).
    """
    df = None
    file.seek(0)
    is_excel = file.name.endswith('.xlsx')
    
    # 1. LETTURA RAW (Senza assumere header o formato)
    if is_excel:
        try:
            df = pd.read_excel(file, header=None)
        except Exception as e:
            st.error(f"Errore Excel: {e}")
            return None
    else:
        # Tentativi CSV (Encoding + Separatori)
        encodings = ['utf-8', 'latin1', 'cp1252']
        separators = [';', ','] 
        
        for enc in encodings:
            for sep in separators:
                try:
                    file.seek(0)
                    # Legge tutto come testo/oggetto per non perdere dati
                    df = pd.read_csv(file, sep=sep, header=None, encoding=enc, engine='python')
                    if df.shape[1] > 1: # Se ha creato colonne, è buono
                        break
                except:
                    continue
            if df is not None: break

    if df is None:
        st.error("Errore fatale: File illeggibile.")
        return None

    try:
        # 2. RICERCA HEADER (Dove inizia la tabella?)
        # Cerca la prima riga che contiene "DATE" o "DATA"
        header_idx = -1
        for i, row in df.iterrows():
            row_str = row.astype(str).str.lower()
            if row_str.str.contains('date').any() or row_str.str.contains('data').any():
                header_idx = i
                break
        
        if header_idx == -1:
            # Fallback: assume riga 0 se non trova nulla
            header_idx = 0
            
        # Promuove la riga trovata a Intestazione
        df.columns = df.iloc[header_idx]
        df = df.iloc[header_idx+1:].reset_index(drop=True)
        # Pulisce nomi colonne
        df.columns = [str(c).strip() for c in df.columns]

        # 3. RILEVAMENTO TRASPOSIZIONE (Il file è ruotato?)
        # Se le colonne (dalla 2a in poi) sembrano date, il file è trasposto
        is_transposed = False
        try:
            sample_cols = df.columns[1:10] # Controlla le prime colonne dati
            valid_dates = 0
            for c in sample_cols:
                try:
                    pd.to_datetime(c, dayfirst=True)
                    valid_dates += 1
                except:
                    pass
            # Se più del 50% delle colonne sono date, è trasposto
            if len(sample_cols) > 0 and (valid_dates / len(sample_cols)) > 0.5:
                is_transposed = True
        except:
            pass

        # 4. NORMALIZZAZIONE STRUTTURA
        if is_transposed:
            # Imposta la prima colonna (Nomi Asset) come indice temporaneo
            df = df.set_index(df.columns[0])
            # Ruota la matrice
            df = df.T
            # Ora l'indice sono le Date, le Colonne sono gli Asset
            df.index.name = 'DATE' # Rinomina indice per standardizzazione
            df = df.reset_index() # Riporta DATE come colonna per il flusso standard

        # 5. PARSING DATA E NUMERI (Logica Standard)
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'data' in col.lower():
                date_col = col
                break
        
        if not date_col:
            st.error("Colonna 'Date' non identificabile.")
            return None

        # Conversione Data
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col])
        df.set_index(date_col, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Via colonne fantasma

        # Conversione Numerica Robusta (Gestisce sia 1.000,00 che 1000.00)
        for col in df.columns:
            series = df[col].astype(str)
            # Tenta conversione standard (US)
            df[col] = pd.to_numeric(series, errors='coerce')
            
            # Se troppi NaN, tenta conversione EU (rimuovi punti, cambia virgola in punto)
            if df[col].isna().sum() > len(df) * 0.5:
                clean_series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(clean_series, errors='coerce')
        
        # Riempimento buchi
        df = df.fillna(method='ffill').dropna()
            
        return df

    except Exception as e:
        st.error(f"Errore elaborazione: {e}")
        return None

def clean_asset_name(name):
    clean = re.sub(r'\s*\(.*\)', '', name)
    return clean.strip()

def get_advanced_stats(weights, returns, annual_factor):
    weights = np.array(weights)
    port_series = returns.dot(weights)
    
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

def optimize_portfolio(returns, annual_factor, min_weight=0.0):
    n_assets = len(returns.columns)
    
    if n_assets * min_weight > 1.0:
        return None 
        
    def objective(weights):
        w = np.array(weights)
        ret = np.sum(returns.mean() * w) * annual_factor
        vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * annual_factor, w)))
        s = ret / vol if vol > 0 else 0
        return -s

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_weight, 1) for _ in range(n_assets))
    init_guess = [1./n_assets for _ in range(n_assets)]
    
    try:
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except:
        return None

@st.cache_data(show_spinner=False)
def find_best_optimized_combination(data, k, annual_factor, max_corr_threshold=1.0, min_w=0.0):
    assets = data.columns.tolist()
    if len(assets) < k: return None, None, (0,0,0,0,0)
    
    best_sharpe = -np.inf
    best_combo = None
    best_weights = None
    best_full_stats = None
    
    if k * min_w > 1.0:
        return None, None, (0,0,0,0,0)
    
    if len(assets) > 15 and k > 2:
        st.warning("⚠️ Troppi asset per calcolo combinatorio completo. Analisi ridotta.")

    for combo in itertools.combinations(assets, k):
        current_corr = get_avg_correlation(data, combo)
        
        if current_corr <= max_corr_threshold:
            subset = data[list(combo)].pct_change().dropna()
            weights = optimize_portfolio(subset, annual_factor, min_weight=min_w)
            
            if weights is not None:
                r, v, s, sort, mdd = get_advanced_stats(weights, subset, annual_factor)
                
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

st.title("🛡️ Quant Allocation: 3-Tier Model")

# SIDEBAR
with st.sidebar:
    st.header("1. Data Feed")
    uploaded_file = st.file_uploader("Carica Dati (CSV o Excel)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("2. Configurazione Dati")
    freq_choice = st.selectbox(
        "Frequenza Dati",
        options=[52, 252, 12],
        index=0, 
        format_func=lambda x: "Settimanale (52)" if x == 52 else ("Giornaliera (252)" if x == 252 else "Mensile (12)")
    )
    annual_factor = freq_choice
    
    manual_placeholder = st.empty()
    
    st.markdown("---")
    st.header("3. Filtri Strategici")
    max_corr_input = st.slider(
        "Max Correlazione Ammessa", 
        min_value=0.0, max_value=1.0, value=1.0, step=0.05
    )
    
    min_weight_pct = st.slider(
        "Peso Minimo per Asset (%)",
        min_value=0, max_value=33, value=10, step=1
    )
    min_weight_val = min_weight_pct / 100.0

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        assets = df.columns.tolist()
        
        with st.spinner('Calcolo Ottimizzazione e Analisi Metodologica...'):
            # 1. Best Single Asset
            temp_sharpes = {}
            for a in assets:
                r_t = df[[a]].pct_change().dropna()
                if not r_t.empty:
                    _, _, s_t, _, _ = get_advanced_stats([1], r_t, annual_factor)
                    temp_sharpes[a] = s_t
                else:
                    temp_sharpes[a] = -999
            
            best_single = max(temp_sharpes, key=temp_sharpes.get)
            
            # UI Manuale
            try:
                default_idx = assets.index(best_single)
            except:
                default_idx = 0
            manual_asset = manual_placeholder.selectbox("2. Linea 1 (Manuale)", assets, index=default_idx)
            
            # Dati Linea 1
            l1_ret_frame = df[[manual_asset]].pct_change().dropna()
            l1_stats = get_advanced_stats([1], l1_ret_frame, annual_factor)
            l1_corr = 1.0
            
            # Floor Peso
            forced_min_w = max(min_weight_val, 0.01)

            # 2. Best Pair
            pair_assets, pair_weights, pair_stats = find_best_optimized_combination(
                df, 2, annual_factor, max_corr_threshold=max_corr_input, min_w=forced_min_w
            )
            if pair_assets:
                l2_corr = get_avg_correlation(df, pair_assets)
                l2_series = df[list(pair_assets)].pct_change().dropna().dot(pair_weights)
            
            # 3. Best Triplet
            triplet_assets, triplet_weights, triplet_stats = find_best_optimized_combination(
                df, 3, annual_factor, max_corr_threshold=max_corr_input, min_w=forced_min_w
            )
            if triplet_assets:
                l3_corr = get_avg_correlation(df, triplet_assets)
                l3_series = df[list(triplet_assets)].pct_change().dropna().dot(triplet_weights)

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ DASHBOARD", "2️⃣ CORRELAZIONI", "3️⃣ BACKTEST", "📘 METODOLOGIA"])

        # --- TAB 1: DASHBOARD ---
        with tab1:
            st.subheader("Allocazione Ottimale")
            
            # Tabella Riepilogativa
            table_data = []
            def make_row(label, asset_list, weights, corr, stats):
                r, v, s, sort, mdd = stats
                if isinstance(asset_list, str): comp_str = f"{clean_asset_name(asset_list)} (100%)"
                else: comp_str = format_composition(asset_list, weights)
                return {
                    "Strategia": label,
                    "Allocazione Sintetica": comp_str,
                    "Corr. Media": f"{corr:.2f}" if isinstance(corr, float) else "N/A",
                    "Rend. Annuo": f"{r*100:.1f}%",
                    "Max DD": f"{mdd*100:.1f}%",
                    "Sharpe": f"{s:.2f}",
                }
            
            table_data.append(make_row("LINEA 1 (Manuale)", manual_asset, [1], l1_corr, l1_stats))
            
            if pair_assets: 
                table_data.append(make_row("LINEA 2 (Best Pair)", pair_assets, pair_weights, l2_corr, pair_stats))
            else: 
                st.warning("LINEA 2: Nessuna combinazione soddisfa i vincoli.")
                
            if triplet_assets: 
                table_data.append(make_row("LINEA 3 (Best Triplet)", triplet_assets, triplet_weights, l3_corr, triplet_stats))
            else:
                 st.warning("LINEA 3: Nessuna combinazione soddisfa i vincoli.")
            
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
            
            st.divider()
            st.markdown("### 📊 Performance vs Rischio")
            col1, col2, col3 = st.columns(3)
            
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
            st.subheader("Matrice di Correlazione")
            unique_assets = list(set([manual_asset] + list(pair_assets or []) + list(triplet_assets or [])))
            clean_labels = {a: clean_asset_name(a) for a in unique_assets}
            fig_corr = px.imshow(df[unique_assets].rename(columns=clean_labels).corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template='plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)

        # --- TAB 3: BACKTEST ---
        with tab3:
            st.subheader("Simulazione Storica")
            common_idx = l1_ret_frame.index
            if pair_assets: common_idx = common_idx.intersection(l2_series.index)
            if triplet_assets: common_idx = common_idx.intersection(l3_series.index)
            
            chart_df = pd.DataFrame(index=common_idx)
            chart_df[f"L1: {clean_asset_name(manual_asset)}"] = (1 + l1_ret_frame.loc[common_idx][manual_asset]).cumprod() * 100
            if pair_assets: chart_df["L2: Best Pair"] = (1 + l2_series.loc[common_idx]).cumprod() * 100
            if triplet_assets: chart_df["L3: Best Triplet"] = (1 + l3_series.loc[common_idx]).cumprod() * 100
            
            fig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, template='plotly_white')
            fig.update_layout(yaxis_title="Valore (Base 100)", legend=dict(orientation="h", y=1.1, title=None))
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 4: METODOLOGIA ---
        with tab4:
            st.markdown("""
            ### Note Tecniche
            * **Flessibilità Totale:** Supporta file Standard e Trasposti (Date in orizzontale).
            * **Pulizia:** Rileva automaticamente l'intestazione anche se ci sono righe vuote.
            * **Numeri:** Supporta formati EU (1.000,00) e US (1000.00).
            """)

    else:
        st.error("File non valido.")
else:
    st.info("Carica il file (CSV o Excel) per iniziare.")
