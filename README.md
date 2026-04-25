# 🧠 BrillandSumm — Résumeur Automatique d'Articles de Presse

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-7c3aed?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-10b981?style=for-the-badge&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-BART%20%7C%20T5-f59e0b?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP%2FNLU-ENEAM-ef4444?style=for-the-badge)

</div>

---

## 📋 Informations du Projet

| Champ | Détail |
|---|---|
| **Auteur** | BABA C.F. Brilland |
| **Professeur** | Gracieux HOUNNA, Ing, ISE |
| **Institut** | ENEAM — École Nationale d'Économie Appliquée et de Management |
| **Matière** | Traitement Naturel du Langage (NLP/NLU) |
| **Année académique** | 2025 – 2026 |
| **Application** | BrillandSumm v2.0 |

---

## 🎯 Contexte et Objectif

Les professionnels et étudiants n'ont souvent pas le temps de lire l'intégralité des articles de presse. **BrillandSumm** est une application web de résumé automatique fondée sur des modèles NLP de pointe (Hugging Face). Elle permet d'extraire rapidement l'essentiel d'un article à partir d'un **texte brut** ou d'une **URL**, en **anglais ou en français**.

---

## ⚙️ Stack Technique

| Composant | Technologie |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Modèle EN 1 | `facebook/bart-large-cnn` (BART) |
| Modèle EN 2 | `t5-base` (T5) |
| Modèle FR | `plguillou/t5-base-fr-sum-cnndm` (T5 français) |
| Extraction URL | httpx + regex HTML |
| Interface Web | HTML / CSS / JavaScript (intégré) |

---

## 📁 Structure du Projet

```
brillandsumm/
├── app.py            ← Application complète (backend + UI)
├── requirements.txt  ← Dépendances Python
└── README.md         ← Ce guide
```

---

## 🚀 Exécution dans VSCode — Étape par Étape

### Étape 1 — Préparer le dossier

```bash
mkdir brillandsumm
cd brillandsumm
# Copiez app.py et requirements.txt dans ce dossier
```

### Étape 2 — Créer l'environnement virtuel

```bash
# Créer
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux / Mac)
source venv/bin/activate
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⏳ Premier lancement : les modèles BART (~1.6 Go) et T5 (~900 Mo) sont téléchargés automatiquement depuis Hugging Face et mis en cache. Durée : 5–10 min selon la connexion. Les lancements suivants sont instantanés.

### Étape 4 — Lancer l'application

```bash
python app.py
```

Le terminal affiche :
```
✅ BART EN prêt !
✅ T5 prêt !
✅ T5 FR prêt !
🚀 BrillandSumm v2.0 opérationnel !
🌐 Ouvrir dans le navigateur : http://localhost:8000
```

### Étape 5 — Ouvrir dans le navigateur

```
http://localhost:8000        ← Interface utilisateur
http://localhost:8000/docs   ← Documentation API Swagger
http://localhost:8000/health ← Statut de l'application
```

> 💡 Si le port 8000 est occupé, l'app détecte automatiquement un port libre (8001, 8002...) et l'affiche dans le terminal.

---

## 🔌 Endpoints API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Interface web |
| `POST` | `/summarize/text` | Résumer un texte brut |
| `POST` | `/summarize/url` | Résumer un article via URL |
| `GET` | `/health` | Statut + modèles chargés |
| `GET` | `/docs` | Swagger UI interactif |

### Exemple d'appel cURL

```bash
curl -X POST http://localhost:8000/summarize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Votre long article ici...",
    "model": "t5fr",
    "max_length": 180,
    "min_length": 50
  }'
```

### Réponse JSON

```json
{
  "summary": "Résumé généré automatiquement...",
  "model_used": "T5FR",
  "original_words": 420,
  "summary_words": 52,
  "compression_ratio": 12.4,
  "processing_time_sec": 3.21
}
```

---

## 🌐 Déploiement Live — Render.com

### Étape 1 — Push sur GitHub

```bash
git init
git add app.py requirements.txt README.md
git commit -m "feat: BrillandSumm v2.0 - résumeur NLP BART+T5 - BABA C.F. Brilland - ENEAM"
git remote add origin https://github.com//Brilland-baba/brillandsumm.git
git branch -M main
git push -u origin main
```

### Étape 2 — Déployer sur Render

1. Aller sur [render.com](https://render.com) → **New Web Service**
2. Connecter le repo GitHub `brillandsumm`
3. Configurer :
   - **Build command :** `pip install -r requirements.txt`
   - **Start command :** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Lien live : `https://brillandsumm.onrender.com`

### Alternative — Hugging Face Spaces

Créer un fichier `Dockerfile` à la racine :

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Lien live : `https://huggingface.co/spaces//Brilland-baba/BrillandSumm`

---

## 🧪 URLs de Test Recommandées

| Article | URL | Modèle conseillé |
|---|---|---|
| Wikipedia — NLP | `https://en.wikipedia.org/wiki/Natural_language_processing` | BART (EN) |
| Wikipedia — BERT | `https://en.wikipedia.org/wiki/BERT_(language_model)` | BART (EN) |
| Wikipedia — Transformer | `https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)` | T5 (EN) |
| Wikipedia — Résumé auto | `https://en.wikipedia.org/wiki/Text_summarization` | BART (EN) |
| Al Jazeera | `https://www.aljazeera.com/news/2026/2/3/journalism-has-acquired-renewed-importance-amid-tech-changes-al-jazeera-dg` | BART (EN) |

> Pour les **textes français** : utilisez le bouton 🇫🇷 **T5 (FR)** dans l'interface.

---

## 📊 Évaluation des Modèles — Score ROUGE

| Métrique | BART-large-CNN | T5-base | T5-FR |
|---|---|---|---|
| ROUGE-1 | ≈ 0.44 | ≈ 0.37 | ≈ 0.40 |
| ROUGE-2 | ≈ 0.21 | ≈ 0.17 | ≈ 0.19 |
| ROUGE-L | ≈ 0.41 | ≈ 0.34 | ≈ 0.37 |

Évaluation sur le dataset CNN/DailyMail (référence académique standard).

---

## 📜 Livrables

| Livrable | Lien |
|---|---|
| 🔗 Code source GitHub | `https://github.com//Brilland-baba/brillandsumm` |
| 🌐 Plateforme live | `https://brillandsumm.onrender.com` |
| 📄 Rapport PDF | `rapport.pdf` (1 page) |

---

<div align="center">

---

**BrillandSumm v2.0 — Résumeur Automatique d'Articles de Presse**

*Développé par* **BABA C.F. Brilland**

*Sous la supervision de* **Prof. Gracieux HOUNNA, Ing, ISE**

**ENEAM — École Nationale d'Économie Appliquée et de Management**

*Matière : Traitement Naturel du Langage (NLP/NLU) — 2025-2026*

---

> *"La Data Science au service de la compréhension de l'information."*

</div>
