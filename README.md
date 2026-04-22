# Image Prompt App (FastAPI + uv)

One-pager FastAPI pour saisir un prompt, generer des images via `gpt-image-2`, les afficher et les sauvegarder automatiquement sur disque.

## Prerequis

- `uv` installe
- Une cle API compatible avec `https://build.lewisnote.com/v1`

## Installation

```bash
uv sync
```

## Variables d'environnement

```bash
export OPENAI_API_KEY="sk-afri-votre-cle"
export OPENAI_BASE_URL="https://build.lewisnote.com/v1"
```

## Lancer l'app

```bash
uv run uvicorn main:app --reload
```

Puis ouvre: `http://127.0.0.1:8000`

## Ce que fait l'app

- Soumission du formulaire en `POST` JSON vers `/api/generate`
- Telechargement automatique des images en local dans `generated/`
- Persistance SQLite dans `data/app.db`
- Sidebar d'historique basee sur les generations precedentes (cover = premiere image)

## Endpoints

- `POST /api/generate`
- `GET /api/history`
- `GET /api/history/{generation_id}`
- `GET /generated/<fichier>`
