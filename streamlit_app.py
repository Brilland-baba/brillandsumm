"""
BrillandSumm v2.0 — Résumeur Automatique d'Articles de Presse
Auteur   : BABA C.F. Brilland
Prof     : Gracieux HOUNNA, Ing, ISE
Institut : ENEAM
Matière  : Traitement Naturel du Langage (NLP/NLU)
"""

# ── Bloquer TensorFlow avant tout import transformers ──────────
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ───────────────────────────────────────────────────────────────

import re
import time
import warnings
warnings.filterwarnings("ignore")

import httpx
import streamlit as st
from transformers import pipeline

# ── Config page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="BrillandSumm — Résumeur Automatique",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS personnalisé ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Fond sombre */
.stApp {
    background: linear-gradient(135deg, #08080f 0%, #0d0820 50%, #080d1a 100%);
    color: #e8e8f0;
}

/* Header principal */
.brilland-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.brilland-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.brilland-title span { -webkit-text-fill-color: #7c3aed; }
.brilland-sub {
    color: #6b7280;
    font-size: 0.88rem;
    margin-top: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Carte résultat */
.result-card {
    background: #10101a;
    border: 1px solid #2a2a3a;
    border-left: 4px solid #7c3aed;
    border-radius: 12px;
    padding: 1.4rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.8;
    color: #e8e8f0;
}

/* Métriques */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-chip {
    background: #0d0d18;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #9ca3af;
}
.metric-chip b { color: #e8e8f0; }

/* Badge modèle */
.model-badge {
    display: inline-block;
    background: #1a0a2e;
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #7c3aed55, transparent);
    margin: 1.5rem 0;
}

/* Signature footer */
.footer-sig {
    text-align: center;
    color: #374151;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #1f1f2e;
    margin-top: 2rem;
}

/* Streamlit overrides */
div[data-testid="stTextArea"] textarea {
    background: #0a0a14 !important;
    border: 1px solid #2a2a3a !important;
    color: #e8e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 10px !important;
}
div[data-testid="stTextInput"] input {
    background: #0a0a14 !important;
    border: 1px solid #2a2a3a !important;
    color: #e8e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 10px !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px #7c3aed33 !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px #7c3aed55 !important;
}
div[data-testid="stSelectbox"] > div {
    background: #0a0a14 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
}
.stSlider > div > div { background: #7c3aed !important; }
div[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Chargement des modèles (cached = chargé UNE seule fois) ────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    models["bart"] = pipeline(
        "summarization", model="facebook/bart-large-cnn",
        device=-1, truncation=True
    )
    models["t5"] = pipeline(
        "summarization", model="t5-base",
        device=-1, truncation=True
    )
    try:
        models["t5fr"] = pipeline(
            "summarization", model="plguillou/t5-base-fr-sum-cnndm",
            device=-1, truncation=True
        )
    except Exception:
        models["t5fr"] = models["t5"]
    return models


# ── Extraction texte depuis URL ─────────────────────────────────
def extract_text_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (BrillandSumm/2.0)"}
    try:
        r = httpx.get(url, headers=headers, timeout=12, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ValueError(f"Impossible de charger l'URL : {e}")
    html = r.text

    if "wikipedia.org" in url:
        m = re.search(r'<div[^>]+id=["\']mw-content-text["\'][^>]*>(.*?)<div[^>]+id=["\']catlinks', html, re.DOTALL)
        if m:
            html = m.group(1)
        html = re.sub(r'<table[^>]*class="[^"]*(?:ambox|navbox|infobox|mbox|sistersitebox)[^"]*"[^>]*>.*?</table>',
                      " ", html, flags=re.DOTALL | re.IGNORECASE)

    for tag in ["script", "style", "nav", "header", "footer", "figure", "aside", "table"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z#0-9]+;", " ", text)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 60]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 80:
        raise ValueError("Texte extrait trop court ou page protégée.")
    return text[:4500]


# ── Résumé ──────────────────────────────────────────────────────
def run_summarizer(models, text, model_key, max_length, min_length):
    input_text = f"summarize: {text}" if model_key in ("t5", "t5fr") else text
    t0 = time.perf_counter()
    result = models[model_key](
        input_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    summary = result[0]["summary_text"]
    return {
        "summary": summary,
        "model_used": model_key.upper(),
        "original_words": len(text.split()),
        "summary_words": len(summary.split()),
        "compression_ratio": round(len(summary.split()) / max(1, len(text.split())) * 100, 1),
        "processing_time_sec": elapsed,
    }


# ════════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="brilland-header">
  <div class="brilland-title">Brilland<span>Summ</span></div>
  <div style="font-size:1rem;color:#9ca3af;margin-top:.3rem">Résumeur Automatique d'Articles de Presse</div>
  <div class="brilland-sub">BABA C.F. Brilland · Prof. Gracieux HOUNNA, Ing, ISE · ENEAM · NLP/NLU</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# Chargement modèles
with st.spinner("⏳ Chargement des modèles BART + T5 + T5-FR… (première fois uniquement)"):
    MODELS = load_models()
st.success("✅ BART · T5 · T5-FR — Tous les modèles sont opérationnels !", icon="🚀")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Sélecteur de modèle ─────────────────────────────────────────
st.markdown("#### 🧠 Choisir le modèle IA")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**BART (EN)**")
    st.caption("facebook/bart-large-cnn\nAnglais — abstractif robuste")
with col2:
    st.markdown("**T5 (EN)**")
    st.caption("t5-base\nAnglais — seq2seq léger")
with col3:
    st.markdown("**🇫🇷 T5 (FR)**")
    st.caption("t5-base-fr-sum-cnndm\nFrançais — fine-tuné ✅")

model_choice = st.selectbox(
    "Modèle sélectionné",
    options=["bart", "t5", "t5fr"],
    format_func=lambda x: {
        "bart": "🧠 BART-large-CNN (Anglais)",
        "t5":   "⚡ T5-base (Anglais)",
        "t5fr": "🇫🇷 T5-FR (Français)"
    }[x],
    label_visibility="collapsed"
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Onglets Texte / URL ─────────────────────────────────────────
tab_text, tab_url = st.tabs(["📝  Texte brut", "🔗  URL d'article"])

# ── TAB TEXTE ───────────────────────────────────────────────────
with tab_text:
    st.markdown("##### Collez votre article")
    input_text = st.text_area(
        "Texte",
        height=200,
        placeholder="Entrez ou collez le texte de l'article à résumer...\n\nConseil : au moins 80 mots pour un bon résumé.",
        label_visibility="collapsed"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        max_len_t = st.slider("Longueur max (tokens)", 60, 300, 180, key="max_t")
    with col_b:
        min_len_t = st.slider("Longueur min (tokens)", 20, 100, 50, key="min_t")

    if st.button("✦ Résumer le texte →", key="btn_text"):
        if not input_text or len(input_text.strip()) < 50:
            st.error("⚠️ Texte trop court. Minimum 50 caractères requis.")
        else:
            with st.spinner("🔄 Génération du résumé en cours…"):
                try:
                    res = run_summarizer(MODELS, input_text[:4500],
                                         model_choice, max_len_t, min_len_t)
                    st.session_state["last_result"] = res
                except Exception as e:
                    st.error(f"⚠️ Erreur : {e}")

# ── TAB URL ─────────────────────────────────────────────────────
with tab_url:
    st.markdown("##### URL de l'article")

    # URLs de test cliquables
    st.markdown("🧪 **URLs de test — cliquer pour copier :**")
    test_urls = {
        "📰 Wikipedia — NLP": "https://en.wikipedia.org/wiki/Natural_language_processing",
        "📰 Wikipedia — Text Summarization": "https://en.wikipedia.org/wiki/Text_summarization",
        "📰 Wikipedia — BERT Model": "https://en.wikipedia.org/wiki/BERT_(language_model)",
        "📰 Wikipedia — Transformer": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "📰 Al Jazeera — AI & Journalism": "https://www.aljazeera.com/news/2026/2/3/journalism-has-acquired-renewed-importance-amid-tech-changes-al-jazeera-dg",
    }

    # Initialiser l'URL dans session_state
    if "url_input" not in st.session_state:
        st.session_state["url_input"] = ""

    cols = st.columns(3)
    url_items = list(test_urls.items())
    for i, (label, url) in enumerate(url_items):
        with cols[i % 3]:
            if st.button(label, key=f"url_chip_{i}"):
                st.session_state["url_input"] = url

    input_url = st.text_input(
        "URL",
        value=st.session_state["url_input"],
        placeholder="https://en.wikipedia.org/wiki/...",
        label_visibility="collapsed",
        key="url_field"
    )

    col_c, col_d = st.columns(2)
    with col_c:
        max_len_u = st.slider("Longueur max (tokens)", 60, 300, 180, key="max_u")
    with col_d:
        min_len_u = st.slider("Longueur min (tokens)", 20, 100, 50, key="min_u")

    if st.button("✦ Résumer l'article →", key="btn_url"):
        if not input_url or not input_url.startswith("http"):
            st.error("⚠️ Veuillez entrer une URL valide (commençant par https://)")
        else:
            with st.spinner("🔄 Extraction du texte et génération du résumé…"):
                try:
                    text = extract_text_from_url(input_url)
                    res = run_summarizer(MODELS, text, model_choice, max_len_u, min_len_u)
                    st.session_state["last_result"] = res
                except Exception as e:
                    st.error(f"⚠️ {e}")

# ── AFFICHAGE RÉSULTAT ──────────────────────────────────────────
if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("#### ✦ Résumé généré")

    st.markdown(f"""
    <div class="model-badge">🧠 Modèle : {r['model_used']}</div>
    <div class="result-card">{r['summary']}</div>
    <div class="metrics-row">
      <div class="metric-chip">📄 Original : <b>{r['original_words']} mots</b></div>
      <div class="metric-chip">✂️ Résumé : <b>{r['summary_words']} mots</b></div>
      <div class="metric-chip">📉 Compression : <b>{r['compression_ratio']}%</b></div>
      <div class="metric-chip">⚡ Temps : <b>{r['processing_time_sec']}s</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.code(r["summary"], language=None)
    st.caption("↑ Cliquez sur l'icône 📋 en haut à droite pour copier le résumé")

# ── FOOTER ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer-sig">
  BrillandSumm v2.0 — Résumeur Automatique d'Articles de Presse<br/>
  BABA C.F. Brilland · ENEAM · NLP/NLU · Prof. Gracieux HOUNNA, Ing, ISE · 2025<br/>
  facebook/bart-large-cnn · t5-base · plguillou/t5-base-fr-sum-cnndm · Hugging Face
</div>
""", unsafe_allow_html=True)
