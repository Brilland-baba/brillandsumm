"""
================================================================================
BrillandSumm v2.0 - Résumeur Automatique d'Articles de Presse
================================================================================

Auteur   : BABA C.F. Brilland
Professeur : Gracieux HOUNNA, Ing, ISE
Institut : ENEAM (École Nationale d'Économie Appliquée et de Management)
Matière  : Traitement Naturel du Langage (NLP/NLU)
Modèles  : BART (facebook/bart-large-cnn) + T5 (t5-base) — Hugging Face

Fonctionnalités :
----------------
1. Résumé de texte brut via API ou interface web
2. Extraction et résumé d'articles depuis une URL
3. Chatbot intelligent pour les requêtes de résumé
4. Interface web moderne avec sélection de modèles
5. Support multilingue (anglais + français avec T5-FR)

Technologies utilisées :
-----------------------
- FastAPI : Framework web asynchrone
- Transformers (Hugging Face) : Modèles NLP pré-entraînés
- Pydantic : Validation des données
- Uvicorn : Serveur ASGI

================================================================================
"""



# ============================================================================
# CONFIGURATION CRITIQUE - Blocage de TensorFlow
# ============================================================================
# Ces variables d'environnement doivent être définies AVANT l'import de transformers
# Pour éviter les conflits entre TensorFlow et PyTorch, on force l'utilisation de PyTorch
import os

# Désactiver complètement TensorFlow

os.environ["USE_TF"] = "0"              # Interdire l'utilisation de TensorFlow
os.environ["USE_TORCH"] = "1"           # Forcer l'utilisation de PyTorch
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Désactiver les modèles TensorFlow dans transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduire les logs de TensorFlow





# ============================================================================
# IMPORTS DES BIBLIOTHÈQUES
# ============================================================================
from fastapi import FastAPI, HTTPException      # Framework web et gestion d'erreurs
from fastapi.responses import HTMLResponse      # Envoi de réponses HTML
from fastapi.middleware.cors import CORSMiddleware  # Gestion du CORS pour l'API
from pydantic import BaseModel                  # Validation des schémas de données
import httpx                                    # Client HTTP pour les requêtes externes
import re                                       # Expressions régulières pour le parsing HTML
import time                                     # Mesure des performances
import warnings                                 # Gestion des avertissements
from transformers import pipeline               # Pipeline Hugging Face pour l'inférence
import uvicorn                                  # Serveur ASGI

# Supprimer les avertissements superflus pour une sortie plus propre
warnings.filterwarnings("ignore")





# ============================================================================
# CHARGEMENT DES MODÈLES (UNE SEULE FOIS AU DÉMARRAGE)
# ============================================================================


print("=" * 80)
print("BrillandSumm v2.0 - Initialisation des modèles NLP")
print("=" * 80)


# Dictionnaire global pour stocker les modèles chargés
MODELS = {}




# ----------------------------------------------------------------------------
# Modèle 1 : BART (Bidirectional and Auto-Regressive Transformer)
# Modèle pré-entraîné sur CNN/DailyMail pour le résumé abstractif en anglais
# Points forts : excellent pour les articles longs, résumé fluide et naturel
# ----------------------------------------------------------------------------


print("⏳ [1/3] Chargement de BART EN (facebook/bart-large-cnn)...")
MODELS["bart"] = pipeline(
    "summarization",                    # Tâche de résumé automatique
    model="facebook/bart-large-cnn",    # Modèle pré-entraîné
    device=-1,                          # -1 = CPU, 0+ = GPU (si disponible)
    truncation=True                     # Troncature automatique des textes longs
)
print("✅ [1/3] BART EN prêt !")




# ----------------------------------------------------------------------------
# Modèle 2 : T5 (Text-to-Text Transfer Transformer)
# Modèle générique de Google, plus léger que BART
# Nécessite le préfixe "summarize: " devant le texte
# ----------------------------------------------------------------------------
print("⏳ [2/3] Chargement de T5 (t5-base)...")
MODELS["t5"] = pipeline(
    "summarization",
    model="t5-base",                    # Version de base du modèle T5
    device=-1,
    truncation=True
)
print("✅ [2/3] T5 prêt !")



# ----------------------------------------------------------------------------
# Modèle 3 : T5 Français (optionnel)
# Version fine-tunée de T5 spécifiquement pour le français
# Si le téléchargement échoue, on utilise T5 anglais comme fallback
# ----------------------------------------------------------------------------


print("⏳ [3/3] Chargement de T5 FR (plguillou/t5-base-fr-sum-cnndm)...")
try:
    MODELS["t5fr"] = pipeline(
        "summarization",
        model="plguillou/t5-base-fr-sum-cnndm",
        device=-1,
        truncation=True
    )
    print("✅ [3/3] T5 FR prêt !")
except Exception as e:
    print(f"⚠️ [3/3] T5 FR non disponible : {e}")
    MODELS["t5fr"] = MODELS["t5"]  # Fallback vers T5 anglais
    print("   → Utilisation de T5 anglais comme alternative")

print("=" * 80)
print("🚀 BrillandSumm v2.0 opérationnel !")
print("   Modèles disponibles : BART EN, T5 EN, T5 FR")
print("=" * 80)
print()




# ============================================================================
# CONFIGURATION DE L'APPLICATION FASTAPI
# ============================================================================



app = FastAPI(
    title="BrillandSumm — Résumeur Automatique d'Articles de Presse",
    description="Résumeur automatique NLP/NLU — BART & T5 FR — ENEAM | BABA C.F. Brilland",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ============================================================================
# SCHÉMAS PYDANTIC (Validation des données)
# ============================================================================



class TextRequest(BaseModel):
    text: str
    model: str = "bart"
    max_length: int = 180
    min_length: int = 50

class UrlRequest(BaseModel):
    url: str
    model: str = "bart"
    max_length: int = 180
    min_length: int = 50

class ChatRequest(BaseModel):
    message: str

# ─────────────────────────────────────────────
# Extraction texte depuis URL
# ─────────────────────────────────────────────

def extract_text_from_url(url: str) -> str:
    """
    Extrait et nettoie le texte principal d'un article à partir de son URL
    
    Paramètres :
    -----------
    url : str
        URL de l'article à analyser
    
    Retourne :
    --------
    str : Texte extrait et nettoyé (max 4500 caractères)
    
    Lève :
    -----
    HTTPException : Si l'URL est inaccessible ou le texte extrait trop court
    
    Processus :
    ----------
    1. Télécharge le contenu HTML de l'URL
    2. Pour Wikipedia : cible spécifiquement la zone de contenu principal
    3. Supprime les balises inutiles (script, style, navigation, etc.)
    4. Nettoie les entités HTML
    5. Filtre les lignes trop courtes (moins de 60 caractères)
    6. Limite à 4500 caractères pour les performances
    """
    # En-têtes HTTP pour simuler un navigateur réel

    headers = {"User-Agent": "Mozilla/5.0 (BrillandSumm/2.0)"}
    try:
        r = httpx.get(url, headers=headers, timeout=12, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Impossible de charger l'URL : {e}")
    html = r.text

    # Extraction intelligente Wikipedia : cibler le contenu principal
    if "wikipedia.org" in url:
        m = re.search(r'<div[^>]+id=["\']mw-content-text["\'][^>]*>(.*?)<div[^>]+id=["\']catlinks', html, re.DOTALL)
        if m:
            html = m.group(1)
        # Supprimer boîtes d'avertissement et tableaux de navigation
        html = re.sub(r'<table[^>]*class="[^"]*(?:ambox|navbox|infobox|mbox|sistersitebox)[^"]*"[^>]*>.*?</table>',
                      " ", html, flags=re.DOTALL | re.IGNORECASE)

    # Nettoyage général des balises inutiles
    for tag in ["script", "style", "nav", "header", "footer", "figure", "aside", "table"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html, flags=re.DOTALL | re.IGNORECASE)

        # Suppression de toutes les balises HTML restantes

    text = re.sub(r"<[^>]+>", " ", html)

    # Remplacement des entités HTML par des espaces

    text = re.sub(r"&[a-z#0-9]+;", " ", text)

    # Filtrage  :: Garder seulement les phrases longues (ignorer menus et labels courts)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 60]
    text = " ".join(lines)

       # Nettoyage final : espaces multiples → un seul espace

    text = re.sub(r"\s+", " ", text).strip()

        # Vérification de la qualité de l'extraction

    if len(text) < 80:
        raise HTTPException(422, "Texte extrait trop court ou page protégée.")
        # Limitation pour les performances des modèles

    return text[:4500]

# ─────────────────────────────────────────────
# Inférence modèle
# ─────────────────────────────────────────────
def run_summarizer(text: str, model_key: str,
                   max_length: int, min_length: int) -> dict:
    

    """
    Exécute le résumé avec le modèle spécifié
    
    Paramètres :
    -----------
    text : str
        Texte source à résumer
    model_key : str
        Clé du modèle ('bart', 't5', 't5fr')
    max_length : int
        Longueur maximale du résumé
    min_length : int
        Longueur minimale du résumé
    
    Retourne :
    --------
    dict : Dictionnaire contenant :
        - summary : Texte du résumé généré
        - model_used : Nom du modèle utilisé
        - original_words : Nombre de mots dans le texte original
        - summary_words : Nombre de mots dans le résumé
        - compression_ratio : Taux de compression (%)
        - processing_time_sec : Temps de calcul en secondes
    
    Notes :
    ------
    - Pour T5, ajoute automatiquement le préfixe "summarize: "
    - Limite la longueur du texte d'entrée pour les performances
    """
    # Vérification que le modèle demandé existe

    if model_key not in MODELS:
        raise HTTPException(400, f"Modèle inconnu : {model_key}. Choisir 'bart' ou 't5'.")

       # Formatage spécial pour T5 : nécessite le préfixe "summarize:"

    input_text = f"summarize: {text}" if model_key in ("t5", "t5fr") else text

       # Mesure des performances

    t0 = time.perf_counter()
    result = MODELS[model_key](
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




# ============================================================================
# ROUTES DE L'API REST
# ============================================================================



@app.post("/summarize/text", summary="Résumer un texte brut (BART ou T5)")
def summarize_text(req: TextRequest):
     

     """
    Endpoint pour résumer un texte fourni directement
    
    Exemple d'utilisation :
    POST /summarize/text
    {
        "text": "Long article text here...",
        "model": "bart",
        "max_length": 180,
        "min_length": 50
    }
    """
    # Validation basique de la longueur du texte

     if len(req.text.strip()) < 50:
        raise HTTPException(422, "Texte trop court (min 50 caractères).")
     return run_summarizer(req.text[:4500], req.model, req.max_length, req.min_length)

@app.post("/summarize/url", summary="Résumer un article via URL (BART ou T5)")
def summarize_url(req: UrlRequest):
    text = extract_text_from_url(req.url)
    return run_summarizer(text, req.model, req.max_length, req.min_length)

@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys()), "app": "BrillandSumm v2.0"}

@app.post("/chat", summary="Chatbot pour résumé")
def chat_endpoint(req: ChatRequest):
     

     """
    Endpoint chatbot qui comprend les commandes naturelles
    
    Commandes supportées :
    ---------------------
    1. "summarize [texte]" - Résume le texte fourni
    2. "summarize url [url]" - Résume l'article à l'URL donnée
    3. Autres messages - Réponse informative du chatbot
    
    Exemples :
    ---------
    - "summarize Natural language processing is a field of AI that focuses..."
    - "summarize url https://en.wikipedia.org/wiki/Transformer"
    - "Hello, what can you do?"
    """
     message = req.message.strip()
     if message.lower().startswith("summarize"):
        parts = message[9:].strip().split(" ", 1)
        if len(parts) > 0 and parts[0].lower() == "url":
            if len(parts) > 1:
                url = parts[1]
                text = extract_text_from_url(url)
                return run_summarizer(text, "bart", 180, 50)
            else:
                return {"response": "Please provide a URL after 'summarize url'."}
        else:
            text = message[9:].strip()
            if text:
                return run_summarizer(text, "bart", 180, 50)
            else:
                return {"response": "Please provide text to summarize after 'summarize'."}
     else:
        return {"response": "Hello! I'm BrillandBot. I can summarize text or articles. Try:\n- summarize [your text]\n- summarize url [url]"}



# ============================================================================
# INTERFACE WEB (HTML/CSS/JS intégré)
# ============================================================================
# L'interface utilisateur est entièrement intégrée dans le code HTML ci-dessous
# Elle offre :
# - Sélection du modèle (BART/T5/T5 FR)
# - Choix du mode (texte brut / URL / chatbot)
# - Ajustement de la longueur du résumé
# - Visualisation des statistiques de compression
# - Mode conversationnel pour les requêtes naturelles



HTML_PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BrillandSumm — Résumeur IA</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#08080f;--card:#10101a;--border:#252535;
  --accent:#7c3aed;--accent2:#a78bfa;--accent3:#60a5fa;
  --text:#e8e8f0;--muted:#6b7280;--green:#10b981;--red:#ef4444;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;
     background-image:
       radial-gradient(ellipse 55% 35% at 75% -5%,#1a0740 0%,transparent 55%),
       radial-gradient(ellipse 45% 30% at 5% 95%,#0d1a3c 0%,transparent 50%)}
header{padding:1.2rem 2.5rem;display:flex;align-items:center;justify-content:space-between;
       border-bottom:1px solid var(--border);backdrop-filter:blur(8px);
       position:sticky;top:0;z-index:100;background:#08080fcc}
.logo{font-size:1.5rem;font-weight:800;letter-spacing:-.03em;
      background:linear-gradient(135deg,#a78bfa 30%,#60a5fa);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo span{-webkit-text-fill-color:#7c3aed}
.hbadges{display:flex;gap:.5rem;align-items:center}
.badge{font-family:'IBM Plex Mono',monospace;font-size:.62rem;
       background:#1a0a2e;border:1px solid #7c3aed44;color:#a78bfa;
       padding:.2rem .6rem;border-radius:999px}
.badge.green{background:#071a0f;border-color:#10b98144;color:#6ee7b7}
main{max-width:900px;margin:2.5rem auto;padding:0 1.5rem}
h1{font-size:2.2rem;font-weight:800;line-height:1.1;margin-bottom:.4rem}
h1 em{font-style:normal;background:linear-gradient(120deg,#a78bfa,#60a5fa);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--muted);font-size:.88rem;margin-bottom:2rem}
.model-bar{display:flex;gap:.6rem;margin-bottom:1.2rem;align-items:center;flex-wrap:wrap}
.model-bar label{font-size:.8rem;color:var(--muted)}
.mchip{padding:.45rem 1.1rem;border-radius:8px;border:1px solid var(--border);
       background:transparent;color:var(--muted);cursor:pointer;
       font-family:'Syne',sans-serif;font-size:.82rem;transition:.15s}
.mchip.active{border-color:var(--accent);color:#fff;
              background:linear-gradient(135deg,#5b21b6,#7c3aed)}
.mchip:hover:not(.active){border-color:var(--accent2);color:var(--accent2)}
.model-desc{font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:var(--muted)}
.tabs{display:flex;gap:.5rem;margin-bottom:1.2rem}
.tab{padding:.5rem 1.3rem;border-radius:8px;border:1px solid var(--border);
     background:transparent;color:var(--muted);cursor:pointer;
     font-family:'Syne',sans-serif;font-size:.88rem;transition:.15s}
.tab.active{background:#60a5fa1a;border-color:var(--accent3);color:var(--accent3)}
.card{background:var(--card);border:1px solid var(--border);
      border-radius:16px;padding:1.6rem;margin-bottom:1.2rem}
textarea,input[type=text]{width:100%;background:#0a0a14;border:1px solid var(--border);
  color:var(--text);border-radius:10px;padding:.9rem 1rem;
  font-family:'IBM Plex Mono',monospace;font-size:.85rem;resize:vertical;outline:none;transition:.2s}
textarea{min-height:140px}
textarea:focus,input[type=text]:focus{border-color:var(--accent);box-shadow:0 0 0 3px #7c3aed1a}
.url-examples{margin-top:.8rem}
.url-examples p{font-size:.75rem;color:var(--muted);margin-bottom:.4rem}
.url-chips{display:flex;gap:.4rem;flex-wrap:wrap}
.url-chip{font-family:'IBM Plex Mono',monospace;font-size:.68rem;
          padding:.3rem .7rem;border-radius:6px;border:1px solid var(--border);
          color:var(--accent2);cursor:pointer;background:#0d0d18;transition:.15s;
          max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.url-chip:hover{border-color:var(--accent2);background:#1a0a2e}
.row{display:flex;gap:1rem;margin-top:1rem;align-items:flex-end;flex-wrap:wrap}
.len-group{display:flex;flex-direction:column;min-width:130px}
.len-group label{font-size:.75rem;color:var(--muted);margin-bottom:.3rem}
input[type=range]{accent-color:var(--accent);width:100%}
.val{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:var(--accent2);margin-top:.15rem}
button.submit{margin-left:auto;padding:.7rem 2rem;
  background:linear-gradient(135deg,#5b21b6,#7c3aed);border:none;border-radius:10px;
  color:#fff;font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;
  cursor:pointer;transition:.2s;display:flex;align-items:center;gap:.5rem;
  box-shadow:0 4px 20px #7c3aed33}
button.submit:hover{transform:translateY(-2px);box-shadow:0 8px 28px #7c3aed55}
button.submit:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
#error{display:none;background:#1f0808;border:1px solid var(--red);border-radius:10px;
       padding:.9rem 1rem;color:#fca5a5;font-size:.88rem;margin-top:.8rem}
#result{display:none}
.result-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:.9rem}
.result-title{font-weight:800;font-size:.95rem;
              background:linear-gradient(135deg,#a78bfa,#60a5fa);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.copy-btn{padding:.35rem .9rem;background:transparent;border:1px solid var(--border);
          color:var(--muted);border-radius:6px;cursor:pointer;font-size:.78rem;transition:.15s}
.copy-btn:hover{border-color:var(--accent2);color:var(--accent2)}
#summary-text{background:#0a0a14;border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:10px;padding:1.2rem;line-height:1.8;font-size:.93rem;
  color:var(--text);white-space:pre-wrap}
.stats{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:.9rem}
.stat{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted);
      background:#0d0d18;border:1px solid var(--border);padding:.3rem .7rem;border-radius:6px}
.stat b{color:var(--text)}
.model-tag{font-family:'IBM Plex Mono',monospace;font-size:.72rem;padding:.3rem .7rem;
           border-radius:6px;background:#1a0a2e;border:1px solid var(--accent)55;color:var(--accent2)}
.spinner{width:16px;height:16px;border:2px solid #ffffff33;border-top-color:#fff;
         border-radius:50%;animation:spin .5s linear infinite;display:none;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
footer{text-align:center;color:var(--muted);font-size:.72rem;padding:2.5rem;
       border-top:1px solid var(--border);font-family:'IBM Plex Mono',monospace;line-height:1.8}
@media(max-width:600px){h1{font-size:1.6rem}header{padding:1rem 1.2rem}.row{flex-direction:column}}
</style>
</head>
<body>
<header>
  <div class="logo">Brilland<span>Summ</span> <em style="-webkit-text-fill-color:#94a3b8;font-style:normal;font-size:.75rem;font-weight:400;letter-spacing:.05em">— Résumeur Automatique</em></div>
  <div class="hbadges">
    <span class="badge">BART · T5</span>
    <span class="badge">Hugging Face</span>
    <span class="badge green">● Live</span>
  </div>
</header>
<main>
  <h1>Résumé <em>Automatique</em><br/>d'Articles de Presse</h1>
  <p class="sub">BABA C.F. Brilland · Prof. Gracieux HOUNNA, Ing, ISE · ENEAM · NLP/NLU</p>

  <div class="model-bar">
    <label>Modèle :</label>
    <button class="mchip active" id="chip-bart" onclick="selectModel('bart')">🧠 BART (EN)</button>
    <button class="mchip" id="chip-t5" onclick="selectModel('t5')">⚡ T5 (EN)</button>
    <button class="mchip" id="chip-t5fr" onclick="selectModel('t5fr')">🇫🇷 T5 (FR)</button>
    <span class="model-desc" id="model-desc">facebook/bart-large-cnn — abstractif anglais</span>
  </div>

  <div class="tabs">
    <button class="tab active" id="tab-btn-text" onclick="switchTab('text')">📝 Texte brut</button>
    <button class="tab" id="tab-btn-url" onclick="switchTab('url')">🔗 URL d'article</button>
    <button class="tab" id="tab-btn-chat" onclick="switchTab('chat')">💬 Chatbot</button>
  </div>

  <div class="card" id="tab-text">
    <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:.5rem">Collez votre article ici</label>
    <textarea id="input-text" placeholder="Entrez ou collez le texte de l'article à résumer...&#10;&#10;Conseil : au moins 80 mots pour un bon résumé."></textarea>
    <div class="row">
      <div class="len-group">
        <label>Longueur max</label>
        <input type="range" id="max-len-t" min="60" max="300" value="180"
               oninput="document.getElementById('mv-t').textContent=this.value"/>
        <div class="val">tokens : <span id="mv-t">180</span></div>
      </div>
      <div class="len-group">
        <label>Longueur min</label>
        <input type="range" id="min-len-t" min="20" max="100" value="50"
               oninput="document.getElementById('nv-t').textContent=this.value"/>
        <div class="val">tokens : <span id="nv-t">50</span></div>
      </div>
      <button class="submit" id="btn-text" onclick="summarizeText()">
        <span class="spinner" id="sp-text"></span>Résumer →
      </button>
    </div>
  </div>

  <div class="card" id="tab-url" style="display:none">
    <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:.5rem">URL de l'article</label>
    <input type="text" id="input-url" placeholder="https://www.bbc.com/news/..."/>
    <div class="url-examples">
      <p>🧪 URLs de test — cliquer pour charger :</p>
      <div class="url-chips">
        <span class="url-chip" onclick="loadUrl('https://en.wikipedia.org/wiki/Natural_language_processing')">📰 Wikipedia — NLP (parfait pour BART)</span>
        <span class="url-chip" onclick="loadUrl('https://en.wikipedia.org/wiki/Text_summarization')">📰 Wikipedia — Text Summarization</span>
        <span class="url-chip" onclick="loadUrl('https://en.wikipedia.org/wiki/BERT_(language_model)')">📰 Wikipedia — BERT Model</span>
        <span class="url-chip" onclick="loadUrl('https://www.aljazeera.com/news/2026/2/3/journalism-has-acquired-renewed-importance-amid-tech-changes-al-jazeera-dg')">📰 Al Jazeera — AI &amp; Journalism 2026</span>
        <span class="url-chip" onclick="loadUrl('https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)')">📰 Wikipedia — Transformer Architecture</span>
      </div>
    </div>
    <div class="row">
      <div class="len-group">
        <label>Longueur max</label>
        <input type="range" id="max-len-u" min="60" max="300" value="180"
               oninput="document.getElementById('mv-u').textContent=this.value"/>
        <div class="val">tokens : <span id="mv-u">180</span></div>
      </div>
      <div class="len-group">
        <label>Longueur min</label>
        <input type="range" id="min-len-u" min="20" max="100" value="50"
               oninput="document.getElementById('nv-u').textContent=this.value"/>
        <div class="val">tokens : <span id="nv-u">50</span></div>
      </div>
      <button class="submit" id="btn-url" onclick="summarizeUrl()">
        <span class="spinner" id="sp-url"></span>Résumer →
      </button>
    </div>
  </div>

  <div class="card" id="tab-chat" style="display:none">
    <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:.5rem">Posez une question ou demandez un résumé</label>
    <div id="chat-messages" style="max-height:300px;overflow-y:auto;margin-bottom:1rem;padding:0.5rem;background:#0a0a14;border:1px solid var(--border);border-radius:10px;"></div>
    <textarea id="input-chat" placeholder="Ex: summarize url https://... ou summarize [texte]" style="min-height:60px;"></textarea>
    <div class="row">
      <button class="submit" id="btn-chat" onclick="sendChat()">
        <span class="spinner" id="sp-chat"></span>Envoyer →
      </button>
    </div>
  </div>

  <div id="error"></div>

  <div class="card" id="result">
    <div class="result-header">
      <span class="result-title">✦ Résumé généré</span>
      <button class="copy-btn" onclick="copyText()">📋 Copier</button>
    </div>
    <div id="summary-text"></div>
    <div class="stats" id="stats"></div>
  </div>
</main>
<footer>
  BrillandSumm v2.0 — Résumeur Automatique d'Articles de Presse<br/>
  BABA C.F. Brilland &nbsp;·&nbsp; ENEAM &nbsp;·&nbsp; NLP/NLU &nbsp;·&nbsp; Prof. Gracieux HOUNNA, Ing, ISE<br/>
  BART-large-CNN &nbsp;·&nbsp; T5-base &nbsp;·&nbsp; T5-FR &nbsp;·&nbsp; Hugging Face Transformers
</footer>

<script>
let currentTab='text', currentModel='bart';
const MODEL_DESC={
  bart:'facebook/bart-large-cnn — abstractif, anglais (CNN/DailyMail)',
  t5:'t5-base — seq2seq léger, anglais, préfixe "summarize:"',
  t5fr:'plguillou/t5-base-fr-sum-cnndm — T5 fine-tuné FRANÇAIS ✅'
};
function selectModel(m){
  currentModel=m;
  ['bart','t5','t5fr'].forEach(k=>document.getElementById('chip-'+k).classList.toggle('active',k===m));
  document.getElementById('model-desc').textContent=MODEL_DESC[m];
}
function switchTab(tab){
  currentTab=tab;
  document.getElementById('tab-text').style.display=tab==='text'?'':'none';
  document.getElementById('tab-url').style.display=tab==='url'?'':'none';
  document.getElementById('tab-chat').style.display=tab==='chat'?'':'none';
  ['text','url','chat'].forEach(t=>{
    const el = document.getElementById('tab-btn-'+t);
    if(el) el.classList.toggle('active',t===tab);
  });
  document.getElementById('result').style.display='none';
  document.getElementById('error').style.display='none';
}
function loadUrl(url){
  document.getElementById('input-url').value=url;
  document.getElementById('input-url').focus();
}
function setLoading(id,on){
  document.getElementById('btn-'+id).disabled=on;
  document.getElementById('sp-'+id).style.display=on?'inline-block':'none';
}
async function doRequest(endpoint,body,btnId){
  setLoading(btnId,true);
  document.getElementById('result').style.display='none';
  document.getElementById('error').style.display='none';
  try{
    const res=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await res.json();
    if(!res.ok)throw new Error(data.detail||'Erreur serveur');
    document.getElementById('summary-text').textContent=data.summary;
    document.getElementById('stats').innerHTML=`
      <span class="model-tag">🧠 ${data.model_used}</span>
      <span class="stat">📄 Original : <b>${data.original_words} mots</b></span>
      <span class="stat">✂️ Résumé : <b>${data.summary_words} mots</b></span>
      <span class="stat">📉 Compression : <b>${data.compression_ratio}%</b></span>
      <span class="stat">⚡ Temps : <b>${data.processing_time_sec}s</b></span>`;
    document.getElementById('result').style.display='block';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'nearest'});
  }catch(e){
    const el=document.getElementById('error');
    el.textContent='⚠ '+e.message;
    el.style.display='block';
  }finally{setLoading(btnId,false);}
}
function summarizeText(){
  const text=document.getElementById('input-text').value.trim();
  if(!text){alert('Veuillez entrer un texte.');return;}
  doRequest('/summarize/text',{text,model:currentModel,
    max_length:+document.getElementById('max-len-t').value,
    min_length:+document.getElementById('min-len-t').value},'text');
}
function summarizeUrl(){
  const url=document.getElementById('input-url').value.trim();
  if(!url){alert('Veuillez entrer ou sélectionner une URL.');return;}
  doRequest('/summarize/url',{url,model:currentModel,
    max_length:+document.getElementById('max-len-u').value,
    min_length:+document.getElementById('min-len-u').value},'url');
}
function copyText(){
  navigator.clipboard.writeText(document.getElementById('summary-text').textContent).then(()=>{
    const b=document.querySelector('.copy-btn');
    b.textContent='✅ Copié !';
    setTimeout(()=>{b.textContent='📋 Copier';},1800);
  });
}
function addMessage(sender, text){
  const div = document.getElementById('chat-messages');
  const msg = document.createElement('div');
  msg.style.marginBottom = '0.5rem';
  msg.style.padding = '0.5rem';
  msg.style.borderRadius = '8px';
  if(sender === 'user'){
    msg.style.background = '#7c3aed1a';
    msg.style.textAlign = 'right';
  }else{
    msg.style.background = '#0d0d18';
  }
  msg.innerHTML = text.replace(/\n/g, '<br>');
  div.appendChild(msg);
  div.scrollTop = div.scrollHeight;
}
async function sendChat(){
  const message = document.getElementById('input-chat').value.trim();
  if(!message){alert('Veuillez entrer un message.');return;}
  addMessage('user', message);
  document.getElementById('input-chat').value = '';
  setLoading('chat', true);
  try{
    const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message})});
    const data = await res.json();
    if(!res.ok) throw new Error(data.detail || 'Erreur');
    if(data.response){
      addMessage('bot', data.response);
    }else{
      addMessage('bot', `✦ Résumé généré<br/><br/><div style="background:#0a0a14;border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:1rem;margin:0.5rem 0;">${data.summary}</div><br/><span style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted);">🧠 ${data.model_used} · 📄 ${data.original_words} mots → ${data.summary_words} mots · 📉 ${data.compression_ratio}% · ⚡ ${data.processing_time_sec}s</span>`);
    }
  }catch(e){
    addMessage('bot', '⚠ Erreur: ' + e.message);
  }finally{
    setLoading('chat', false);
  }
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
        """
    Point d'entrée principal - Interface utilisateur web
    Retourne la page HTML complète avec l'interface de résumé
    """
        return HTML_PAGE



# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================


if __name__ == "__main__":
    """
    Lancement du serveur Uvicorn avec configuration optimisée
    
    Configuration :
    --------------
    - Host: 0.0.0.0 (écoute sur toutes les interfaces réseau)
    - Port: Recherche automatique d'un port libre entre 8000 et 8009
    - Reload: Désactivé pour les performances (réactiver en développement)
    - Workers: 1 (optimisé pour les modèles chargés en mémoire)
    """
    import socket

        # Recherche automatique d'un port disponible

    port = 8000
    for p in range(8000, 8010):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", p)) != 0:
                port = p
                break
    
    
    
    print("\n" + "=" * 80)
    print("🚀 BRILLANDSUMM V2.0 - DÉMARRAGE DU SERVEUR")
    print("=" * 80)
    print(f"📡 Serveur HTTP : http://localhost:{port}")
    print(f"🔗 Interface web : http://localhost:{port}/")
    print(f"📚 Documentation API : http://localhost:{port}/docs")
    print(f"🩺 Vérification santé : http://localhost:{port}/health")
    print("\n⚙️ Configuration :")
    print(f"   • Modèles chargés : {len(MODELS)}")
    print(f"   • Modèles disponibles : BART, T5, T5 Français")
    print(f"   • Platforme : CPU (device=-1)")
    print("\n💡 Commandes utiles :")
    print(f"   • CTRL+C : Arrêter le serveur")
    print(f"   • Ouvrir http://localhost:{port} dans le navigateur")
    print("\n📝 Notes :")
    print("   • Premier chargement peut prendre 20-30 secondes")
    print("   • T5 FR peut nécessiter un téléchargement initial")
    print("=" * 80)
    print()
    
    # Lancement du serveur
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        reload=False,      # Off pour production (meilleures performances)
        workers=1          # 1 worker car modèles lourds en mémoire
    )