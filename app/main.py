import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import textwrap
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'creditcard.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'fraud_model.pkl')

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Fraud Monitor",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# --- SIDEBAR CONTACT ---
with st.sidebar:
    st.markdown("## Sofiane El Morabit")
    st.markdown("Business/Digital/Data Analyst")
    st.markdown("---")
    st.markdown("👉 [Mon Profil LinkedIn](https://www.linkedin.com/in/sofiane-el-morabit-4a71aa20b//)")
    st.markdown("👉 [Mon Portfolio GitHub](https://github.com/sofianee9/)")
    st.markdown("---")
    st.markdown("**Tech Stack :**")
    st.markdown(
        "- Python (Pandas, scikit-learn)\n"
        "- Scikit-learn (Modélisation + Résolution imbalance)\n"
        "- Plotly Express (Data Viz interactive)\n"
        "- Streamlit (App & Simulation temps réel)"
    )

# --- CSS GLOBAL ---
st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; }
        .block-container { padding-top: 5rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #0F172A !important; font-family: sans-serif; font-weight: 700; }
        div[data-testid="stMetric"] { display: none; }
        div.stAlert { display: none; }

        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
        }
        [data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }
        [data-testid="stSidebar"] a {
            color: #60A5FA !important;
            text-decoration: none;
        }
        [data-testid="stSidebar"] a:hover {
            text-decoration: underline;
        }
        
        div.stButton > button {
            background-color: #0F172A;
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
            margin-top: 0px !important;
        }
        div.stButton > button:hover {
            background-color: #1E293B;
            color: #FFFFFF !important;
            border: 1px solid #3B82F6;
        }
        
        div[data-testid="column"] {
            align-self: start; 
        }
    </style>
""", unsafe_allow_html=True)

# --- FONCTION DE CHARGEMENT ---
@st.cache_data
def load_data_and_metrics():
    if not os.path.exists(DATA_PATH): return None, None, None, None, None
    df = pd.read_csv(DATA_PATH, nrows=150000)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if not os.path.exists(MODEL_PATH): return None, None, None, None, None
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return df, X_test, y_test, y_pred, y_proba, model

# --- EXÉCUTION ---
df_full, X_test, y_test, y_pred, y_proba, model = load_data_and_metrics()

if df_full is None:
    st.error("❌ Erreur : Données ou Modèle introuvable.")
    st.stop()

# --- HEADER ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 40px;">
        <div style="display: flex; align-items: center; justify-content: center;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#0F172A" width="60px" height="60px">
                <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
            </svg>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <h1 style="color: #0F172A; font-family: sans-serif; font-weight: 900; font-size: 40px; margin: 0; line-height: 1.1;">
                Fraud Monitor
            </h1>
            <p style="margin: -5px 0 0 0; color: #64748B; font-size: 16px; font-weight: 500;">
                Plateforme d'Audit et de Détection des Risques Financiers
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CONTEXTE & OBJECTIFS ---
st.markdown("""
<div style="background-color:#FFFFFF; border-radius:10px; padding:18px 22px; 
            border:1px solid #E2E8F0; margin-bottom:25px;">
  <h3 style="margin-top:0; color:#0F172A;"> Contexte & Objectifs</h3>
  <p style="color:#475569; font-size:14px; margin-bottom:8px;">
    Ce tableau de bord s'appuie sur un jeu de données réel de transactions par carte bancaire 
    (dataset Kaggle <i>Credit Card Fraud Detection</i>), fortement déséquilibré : les fraudes 
    représentent une proportion infime des opérations.
  </p>
  <p style="color:#475569; font-size:14px; margin-bottom:8px;">
    L'objectif est de simuler un système de scoring IA capable d'identifier les transactions 
    suspectes avant le débit, tout en limitant au maximum les blocages injustifiés.
  </p>
  <ul style="color:#475569; font-size:14px; margin-top:6px; margin-bottom:0;">
    <li><b>Détecter un maximum de fraudes</b> (priorité au Recall).</li>
    <li><b>Réduire les fuites critiques</b> (faux négatifs) qui génèrent des pertes financières.</li>
    <li><b>Fournir une vue claire au métier</b> : matrice de détection, audit synthétique, interprétabilité des variables.</li>
    <li><b>Proposer un simulateur temps réel</b> pour tester le modèle sur des transactions individuelles.</li>
  </ul>
</div>
""", unsafe_allow_html=True)


# --- CALCULS ---
fraud_rate = (df_full['Class'].sum() / len(df_full)) * 100
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# --- FONCTION CARTE KPI ---
def display_kpi_card(title, value, tooltip_text):
    tooltip_html = f'<span title="{tooltip_text}" style="color: #64748B; cursor: help; font-size: 16px; font-weight: bold; margin-left: 8px;">?</span>'
    html = f"""<div style="background-color: #0F172A; padding: 20px; border-radius: 10px; border: 1px solid #1E293B; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px; min-height: 120px; display: flex; flex-direction: column; justify-content: center;"><div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="color: #F8FAFC; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">{title}</span>{tooltip_html}</div><div style="color: #FFFFFF; font-size: 28px; font-weight: 700;">{value}</div></div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- FONCTION CARTE AUDIT ---
def display_audit_card(title, value, subtext, is_highlight=False):
    border_color = "#3B82F6" if is_highlight else "#1E293B"
    html = f"""<div style="background-color: #0F172A; padding: 15px 20px; border-radius: 10px; border: 1px solid {border_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 10px; display: flex; flex-direction: column; justify-content: center;"><div style="display: flex; align-items: center; margin-bottom: 5px;"><span style="color: #F8FAFC; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">{title}</span></div><div style="color: #FFFFFF; font-size: 22px; font-weight: 700; margin-bottom: 5px;">{value}</div><div style="color: #94A3B8; font-size: 11px; font-style: italic;">{subtext}</div></div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- AFFICHAGE KPIS ---
k1, k2, k3, k4 = st.columns(4)
with k1: display_kpi_card("Transactions", f"{len(df_full):,}", "Nombre total de transactions analysées.")
with k2: display_kpi_card("Taux de Risque", f"{fraud_rate:.3f} %", "Pourcentage de transactions frauduleuses.")
with k3: display_kpi_card("Sécurité (Recall)", f"{recall:.1%}", "Capacité à détecter les VRAIES fraudes.")
with k4: display_kpi_card("Fiabilité (AUC)", f"{roc_auc:.3f}", "Score de performance globale.")

st.markdown("###")

# --- VISUALISATIONS ---
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("🛡️ Matrice de Détection")
    matrix_data = [[tn, fp], [fn, tp]]
    
    fig = px.imshow(
        matrix_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, '#F1F5F9'], [1, '#0F172A']], 
        x=['Prédit: OK', 'Prédit: FRAUDE'],
        y=['Réel: OK', 'Réel: FRAUDE']
    )
    fig.update_traces(texttemplate="%{z}", textfont_size=24, textfont_weight="bold")
    fig.add_annotation(x=0, y=1, text="⚠️ RATÉS", showarrow=False, yshift=30, font=dict(color="#DC2626", size=14, family="Arial Black"))
    fig.add_annotation(x=1, y=1, text="✅ BLOQUÉS", showarrow=False, yshift=30, font=dict(color="#10B981", size=14, family="Arial Black"))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A"),
        margin=dict(t=30, l=10, r=10, b=10),
        height=400,
        coloraxis_showscale=False
    )
    fig.update_xaxes(title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    fig.update_yaxes(title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("📑 Rapport d'Audit")
    display_audit_card("✅ Interceptions", f"{tp}", "Transactions bloquées avant débit. Protection directe de la trésorerie.")
    display_audit_card("🚨 Fuites Critiques", f"{fn}", "Faux Négatifs : Fraudes passées sous le radar. Risque de chargeback.", is_highlight=True)
    display_audit_card("📈 Stratégie IA", f"{recall:.1%}", "Optimisation Sécurité : Le modèle privilégie la détection pour stopper l'hémorragie financière.")

st.markdown("###")

# --- SECTION EXPLICABILITÉ ---
st.subheader("🔍 Comprendre la Décision de l'IA")

st.markdown("""
<div style="background-color: #1E293B; padding: 15px; border-left: 5px solid #3B82F6; border-radius: 5px; margin-bottom: 20px;">
    <p style="color: #F1F5F9; margin: 0; font-size: 14px;">
        <b>🧠 Décryptage :</b> Ce graphique révèle la logique interne du modèle. 
        Les barres indiquent les variables techniques (anonymisées V1-V28) qui pèsent le plus dans la balance pour classer une transaction comme <b>Fraude</b> (Rouge) ou <b>Légitime</b> (Bleu).
    </p>
</div>
""", unsafe_allow_html=True)

coeffs = pd.DataFrame({'Feature': list(X_test.columns), 'Weight': model.coef_[0]})
coeffs['Abs_Weight'] = coeffs['Weight'].abs()
top_features = coeffs.sort_values(by='Abs_Weight', ascending=False).head(10)

fig_feat = px.bar(
    top_features,
    x='Weight',
    y='Feature',
    orientation='h',
    labels={'Weight': 'Impact (Coefficients)', 'Feature': 'Variable'},
    color='Weight',
    color_continuous_scale="RdBu_r" 
)

fig_feat.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#0F172A"),
    height=400,
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False
)
fig_feat.update_xaxes(title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
fig_feat.update_yaxes(title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))

st.plotly_chart(fig_feat, use_container_width=True)

# --- SIMULATEUR ---
st.markdown("---")
st.subheader("🤖 Simulateur d'IA en Temps Réel")

sim_btn, sim_res = st.columns([1, 3])

with sim_btn:
    if st.button("🎲 Analyser une transaction", use_container_width=True):
        random_tx = X_test.sample(1)
        pred = model.predict(random_tx)[0]
        prob = model.predict_proba(random_tx)[0][1]
        st.session_state['last_pred'] = {"pred": pred, "prob": prob, "data": random_tx}
        st.rerun()

with sim_res:
    if 'last_pred' in st.session_state:
        res = st.session_state['last_pred']
        pct = res['prob'] * 100
        
        if res['pred'] == 1:
            st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 1px solid #DC2626; padding: 15px; border-radius: 10px; color: #991B1B; margin-bottom: 15px;">
                <h3 style="margin:0; color: #DC2626; font-size: 20px;">🚨 ALERTE FRAUDE</h3>
                <p style="margin: 5px 0 0 0;">Probabilité de Risque : <b>{pct:.2f}%</b> | Transaction Bloquée</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #F0FDF4; border: 1px solid #16A34A; padding: 15px; border-radius: 10px; color: #166534; margin-bottom: 15px;">
                <h3 style="margin:0; color: #16A34A; font-size: 20px;">✅ Transaction Légitime</h3>
                <p style="margin: 5px 0 0 0;">Score de Sûreté : <b>{100-pct:.2f}%</b> | Transaction Validée</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<h4 style='color: #0F172A; margin-top: 20px;'>🔍 Données Techniques (V1-V28) :</h4>", unsafe_allow_html=True)
        st.dataframe(res['data'], hide_index=True, use_container_width=True)
        
    else:
        st.info("👆 Cliquez sur le bouton 'Analyser' pour simuler l'arrivée d'une nouvelle transaction.")