"""
app.py
------
Streamlit dashboard for the IT Support Ticket Classifier.

Run with:
    streamlit run app.py
"""

import re, os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import clean_text, META_COLS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(__file__)
MDL  = os.path.join(BASE, "models")
OUT  = os.path.join(BASE, "outputs")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #457b9d, #2a9d8f);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
    }
    .kpi-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        border-left: 4px solid #457b9d;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #457b9d; }
    .kpi-label { font-size: 0.85rem; color: #6c757d; margin-top: 0.2rem; }
    .pred-box {
        border-radius: 12px; padding: 1.5rem;
        border: 2px solid #dee2e6; margin-top: 1rem;
    }
    .high-badge   { background:#fde8ea; color:#e63946; padding:6px 16px;
                    border-radius:20px; font-weight:700; font-size:1rem; }
    .medium-badge { background:#fff3e0; color:#f4a261; padding:6px 16px;
                    border-radius:20px; font-weight:700; font-size:1rem; }
    .low-badge    { background:#e8f5e9; color:#2a9d8f; padding:6px 16px;
                    border-radius:20px; font-weight:700; font-size:1rem; }
    .stButton>button {
        background: linear-gradient(135deg, #457b9d, #2a9d8f);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    required = ["tfidf.pkl","scaler.pkl","cat_model.pkl",
                 "pri_model.pkl","le_cat.pkl","le_pri.pkl"]
    missing  = [f for f in required if not os.path.exists(os.path.join(MDL, f))]
    if missing:
        return None
    return {
        "tfidf"  : joblib.load(os.path.join(MDL, "tfidf.pkl")),
        "scaler" : joblib.load(os.path.join(MDL, "scaler.pkl")),
        "cat"    : joblib.load(os.path.join(MDL, "cat_model.pkl")),
        "pri"    : joblib.load(os.path.join(MDL, "pri_model.pkl")),
        "le_cat" : joblib.load(os.path.join(MDL, "le_cat.pkl")),
        "le_pri" : joblib.load(os.path.join(MDL, "le_pri.pkl")),
    }


@st.cache_data
def load_summary():
    path = os.path.join(OUT, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset():
    path = os.path.join(BASE, "Data", "tickets_clean.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Prediction logic ──────────────────────────────────────────────────────────
def featurise(text, models, include_meta=True):
    cleaned = clean_text(text)
    x_tfidf = models["tfidf"].transform([cleaned])
    if not include_meta:
        return x_tfidf
    meta = {
        "char_len"    : len(text),
        "word_count"  : len(text.split()),
        "exclaim"     : text.count("!"),
        "question"    : text.count("?"),
        "urgency_flag": int(bool(re.search(
            r"\b(?:urgent|asap|immediately|critical|emergency|broken|crash|"
            r"production|hack|compromised|down|outage)\b",
            text.lower()))),
    }
    x_meta = models["scaler"].transform(
        csr_matrix(pd.DataFrame([meta])[META_COLS].values))
    return hstack([x_tfidf, x_meta])


def predict(text, models):
    x_cat = featurise(text, models, include_meta=False)
    x_pri = featurise(text, models, include_meta=True)
    category = models["le_cat"].inverse_transform(models["cat"].predict(x_cat))[0]

    if hasattr(models["pri"], "predict_proba"):
        prob = models["pri"].predict_proba(x_pri)[0]
    else:
        raw  = models["pri"].decision_function(x_pri)[0]
        prob = np.exp(raw) / np.exp(raw).sum()

    priority   = models["le_pri"].inverse_transform(models["pri"].predict(x_pri))[0]
    cat_conf   = float(prob.max())

    if hasattr(models["cat"], "predict_proba"):
        cat_prob = models["cat"].predict_proba(x_cat)[0].max()
    elif hasattr(models["cat"], "decision_function"):
        dv = models["cat"].decision_function(x_cat)[0]
        cat_prob = float(np.exp(dv.max()) / np.exp(dv).sum())
    else:
        cat_prob = 1.0

    return category, priority, float(cat_prob), float(cat_conf)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎫 Ticket Classifier")
    st.markdown("---")
    page = st.radio("Navigate", ["🔮 Predict", "📊 Dashboard", "📖 About"])
    st.markdown("---")
    summary = load_summary()
    if summary:
        st.markdown("**Model Info**")
        st.caption(f"Category model: {summary['cat_model']}")
        st.caption(f"Priority model: {summary['pri_model']}")
        st.caption(f"Training tickets: {summary['n_train']:,}")
        st.caption(f"Features: {summary['n_features']:,}")

models = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.markdown('<p class="main-header">🎫 Support Ticket Classifier</p>',
                unsafe_allow_html=True)
    st.markdown("Type or paste a support ticket below and the ML model will "
                "classify it instantly.")

    if models is None:
        st.error("⚠️ Models not found. Please run `python train.py` first.")
        st.stop()

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### ✍️ Enter Ticket Details")
        st.markdown("#### 🧪 Or try a sample ticket")
        samples = {
            "— choose a sample —": ("", ""),
            "Computer-Services — Printer setup": (
                "Instalar impressora",
                "Instalar impressoraBoa tarde.  Gostaria de instalar em meu computador uma impressora.  Impressora RENA. [COMPANY]elo Enveloper imager 2.5.  É uma impressora para imprimir em pacotinhos de sementes.      Obrigado!",
            ),
            "Eol — End-of-Life system check": (
                "The following system must be set to End Of Life",
                "The following system must be set to End Of Life in Nexthink and checked for[LOCATION]reservations:  [SERVER]  -Automated ticket creation-Please remove the system from Nexthink according to the[LOCATION]process and please verify if any[LOCATION]reservations exist. If yes. please remove reservations and note down the corresponding IP(s) and inform [COMPANY] network team to remove firewall rules if there are any.",
            ),
            "Fileservice — File share access": (
                "File Share Access",
                "File Share Access - [TICKET ID] - [NAME] ([COMPANY]. [LOCATION])Service Request: File Share Access     Grant access to file share for the following user:  [USERNAME] ([COMPANY]. [LOCATION]) (Username: [USERNAME])  Access Type: (RW) Read / Write  File Share:  J:/     Requestor: [NAME] ([COMPANY]. [LOCATION])  Comment:  [NAME] is our new employee in [ADDRESS]. Could you please create her an access to our file share?  Thank you!",
            ),
            "Support General — Server HTTPS access": (
                "Zugriff auf Server",
                "Zugriff auf[LOCATION]Server von einem anderen[LOCATION]ServerHallo Zusammen.      vom [IP] gibt es keinen keinen Zugriff auf den Port 443 (https) des [IP]. Die andere Richtung funktioniert. Für das Geoprocessing (eine Webapplikation) brauchen wir aber eine beidseitige Kommunikation auf die https-Ports des Applikationservers und des Geoprocessing-Servers.      Können Sie bitte den Zugriff vom IP] auf den Port 443 (https) des [IP]  freischalten?                Danek udn Viele Grüße  [NAME]",
            ),
            "Software — Nautilus installation": (
                "Need Nautilus and the Nautilus Citrix app",
                "Need Nautilus and the Nautilus Citrix appI need the nautilus 9.3 software and the Nautilus 9.3 Citrix app installed on my PC.  This may have to go through [COMPANY] for approval.  Please contact [NAME] if you need further information.",
            ),
            "O365 — Teams and Skype integration": (
                "Teams & Skype integration question",
                "Teams & Skype integration questionCan someone look at integration problems between Teams and Skype?  Example:  I am online in Teams. but not in Skype now. When I try to contact someone (chat or call) that is Online in Skype but not in Teams. it doesnt work. Is this by design?  thanks    [NAME]BA. PMP. CISSP   [COMPANY] Services North America  Phone [PHONE]         Mobile [PHONE]  [EMAIL]<mailto:[EMAIL]>            [URL]>",
            ),
            "Active Directory — User deactivation": (
                "Cancelamento Usuário e Acessos",
                "Cancelamento Usuário e Acessos [NAME]Por gentileza cancelar o usuário da [NAME] e criar o aliás no e-mail para [NAME] (segue anexo autorização).  Obrigada!",
            ),
        }

        if "ticket_subject" not in st.session_state:
            st.session_state.ticket_subject = ""
        if "ticket_body" not in st.session_state:
            st.session_state.ticket_body = ""

        def load_sample():
            choice = st.session_state.sample_choice
            if choice != "— choose a sample —":
                sample_subject, sample_body = samples[choice]
                st.session_state.ticket_subject = sample_subject
                st.session_state.ticket_body = sample_body

        st.selectbox(
            "Sample tickets",
            list(samples.keys()),
            key="sample_choice",
            on_change=load_sample,
        )

        subject = st.text_input(
            "Subject line",
            key="ticket_subject",
            placeholder="e.g. App keeps crashing on startup",
        )
        body = st.text_area(
            "Ticket body",
            key="ticket_body",
            height=160,
            placeholder="Describe the issue in detail…",
        )

        predict_btn = st.button("🔮 Classify Ticket")

    with col2:
        st.markdown("#### 📋 Prediction Result")
        if predict_btn and (subject.strip() or body.strip()):
            full_text = subject + " " + body
            with st.spinner("Classifying…"):
                category, priority, cat_conf, pri_conf = predict(full_text, models)

            pri_badge = {
                "High":   '<span class="high-badge">🔴 HIGH</span>',
                "Medium": '<span class="medium-badge">🟡 MEDIUM</span>',
                "Low":    '<span class="low-badge">🟢 LOW</span>',
            }.get(priority, priority)

            st.markdown('<div class="pred-box">', unsafe_allow_html=True)
            st.markdown(f"**Category**")
            st.markdown(f"### 📂 {category}")
            st.markdown(f"**Priority**")
            st.markdown(pri_badge, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Category Relative Score", f"{cat_conf*100:.1f}%")
            c2.metric("Priority Relative Score", f"{pri_conf*100:.1f}%")

            # Routing suggestion
            routing = {
                "Support General": "General IT Support",
                "Fileservice": "File Services Support",
                "Software": "Software Support",
                "O365": "Microsoft 365 Support",
                "Active Directory": "Identity and Access Management",
                "Computer-Services": "Computer Services / Endpoint Support",
                "Eol": "End-of-Life / Asset Lifecycle Support",
            }
            team = routing.get(category, "📋 General Support")
            st.markdown(f"**Route to:** {team}")
            st.markdown('</div>', unsafe_allow_html=True)

        elif predict_btn:
            st.warning("Please enter a subject or body text.")
        else:
            st.info("Fill in the ticket details and click **Classify Ticket**.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown('<p class="main-header">📊 Model Performance Dashboard</p>',
                unsafe_allow_html=True)

    summary = load_summary()
    df      = load_dataset()

    if summary is None:
        st.error("⚠️ No training results found. Run `python train.py` first.")
        st.stop()

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{summary['cat_accuracy']*100:.1f}%</div>
            <div class="kpi-label">Category Accuracy</div></div>""",
            unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{summary['pri_accuracy']*100:.1f}%</div>
            <div class="kpi-label">Priority Accuracy</div></div>""",
            unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{summary['n_train']:,}</div>
            <div class="kpi-label">Training Tickets</div></div>""",
            unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{len(summary['categories'])}</div>
            <div class="kpi-label">Categories</div></div>""",
            unsafe_allow_html=True)

    if "cat_macro_f1" in summary and "pri_macro_f1" in summary:
        m1, m2 = st.columns(2)
        m1.metric("Category Macro F1", f"{summary['cat_macro_f1']*100:.1f}%")
        m2.metric("Priority Macro F1", f"{summary['pri_macro_f1']*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts from outputs/
    chart_pairs = [
        ("cat_confusion_matrix.png",  "pri_confusion_matrix.png"),
        ("cat_f1_per_class.png",      "pri_f1_per_class.png"),
        ("cat_model_comparison.png",  "pri_model_comparison.png"),
    ]
    titles = [
        ("Category — Confusion Matrix", "Priority — Confusion Matrix"),
        ("Category — F1 per Class",     "Priority — F1 per Class"),
        ("Category — Model Comparison", "Priority — Model Comparison"),
    ]

    for (lf, rf), (lt, rt) in zip(chart_pairs, titles):
        lp = os.path.join(OUT, lf)
        rp = os.path.join(OUT, rf)
        if os.path.exists(lp) and os.path.exists(rp):
            c1, c2 = st.columns(2, gap="medium")
            c1.markdown(f"**{lt}**")
            c1.image(lp, use_container_width=True)
            c2.markdown(f"**{rt}**")
            c2.image(rp, use_container_width=True)
            st.markdown("---")

    # Dataset overview
    if df is not None:
        st.markdown("#### 📂 Dataset Overview")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            cc = df["category"].value_counts()
            colors = ["#457b9d","#e63946","#f4a261","#2a9d8f","#8338ec","#06d6a0","#ffb703"]
            ax.barh(cc.index, cc.values, color=colors[:len(cc)], alpha=0.88)
            ax.invert_yaxis()
            ax.set_title("Tickets per Category", fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            c1.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            pc = df["priority"].value_counts().reindex(["High","Medium","Low"])
            PRI_COL = {"High":"#e63946","Medium":"#f4a261","Low":"#2a9d8f"}
            ax.pie(pc.values, labels=pc.index, autopct="%1.1f%%",
                   colors=[PRI_COL[p] for p in pc.index],
                   startangle=140, wedgeprops=dict(edgecolor="white",linewidth=2))
            ax.set_title("Priority Distribution", fontweight="bold")
            plt.tight_layout()
            c2.pyplot(fig)
            plt.close()

        st.markdown("#### 🗂️ Sample Tickets")
        st.dataframe(df[["text","category","priority"]].sample(8, random_state=1)
                       .reset_index(drop=True), height=260)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 About":
    st.markdown('<p class="main-header">📖 About This Project</p>',
                unsafe_allow_html=True)

    st.markdown("""
## What This Project Does

This ML system reads raw IT support ticket text and automatically:
1. **Classifies** it into one of seven dataset categories
2. **Predicts a rule-derived priority label** (High / Medium / Low)

This helps support teams route tickets instantly and focus on urgent issues first.

---

## Dataset

**Zenodo IT Support Ticket Dataset**  
- 2,229 real tickets manually classified by 3 IT professionals  
- 7 categories: Support General, Fileservice, Software, O365, Active Directory, Computer-Services, Eol
- Multilingual: English, German, Portuguese, Spanish  
- All PII removed and replaced with tags like `[NAME]`, `[EMAIL]`  
- Source: https://zenodo.org/records/7648117

---

## ML Pipeline

| Step | What Happens |
|------|-------------|
| Text Cleaning | Lowercase, remove PII tags, URLs, punctuation, stopwords |
| TF-IDF Vectorisation | Training-set-only TF-IDF using 1+2-grams |
| Meta Features | Ticket length, exclamation marks, urgency keyword flag |
| Category Model | Best of Logistic Regression, Linear SVC, Naive Bayes, Random Forest |
| Priority Model | Best of the same classifiers, trained on rule-derived labels |
| Model Selection | Cross-validation with macro-F1; final held-out supplied test set |
| Data Leakage | TF-IDF and scaling are fitted only on the supplied training split |

---

## Why Priority Accuracy Is Lower

Priority is **not in the original dataset** — we engineer it from keyword rules.  
These are **rule-derived heuristic labels**, not verified human-labelled business urgency.
Use the priority output as a transparent routing signal, not a verified service-level decision.

**Score interpretation:** Category and priority relative scores are independent model scores.  
Probabilities are displayed only when the selected model provides `predict_proba`; otherwise, the UI labels them as relative model scores and does not call them calibrated confidence.

---

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `Streamlit` · `joblib`


    """)
