# Briefly — Extractive Text Summarizer

A local-first web app that creates concise summaries by selecting the most informative sentences from a document. It uses TF-IDF features and a Random Forest classifier; no external API or LLM is required.

## Features

- Summarize pasted text or TXT, PDF, and DOCX files.
- Process English and Ukrainian text. The trained model is English-only; Ukrainian uses local TF-IDF features and has not been evaluated.
- Choose a sentence count or a target reduction.
- Compare Random Forest with Lead-N and TF-IDF baselines using ROUGE metrics.
- Train and evaluate models on BBC News Summary or the included demo dataset.

## Quick start

### Docker

```bash
docker compose up --build
```

Open the app at <http://localhost:3000>. The API and Swagger UI are at <http://localhost:8000> and <http://localhost:8000/docs>.

### Local development

Requires Python 3.12+ and Node.js 20+.

In PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current window, then activate the environment again.

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Vite serves the frontend at <http://localhost:3000> and proxies `/api` requests to the backend.

On macOS or Linux, activate the virtual environment with `source .venv/bin/activate`.

## Data and model

The repository includes 60 BBC News Summary demo article-summary pairs, a pretrained model, and experiment artifacts, so the app can run without downloading a dataset.

For full training and evaluation, place BBC News Summary in this structure:

```text
backend/data/bbc/
  News Articles/<category>/<id>.txt
  Summaries/<category>/<id>.txt
```

CSV datasets are also supported. From `backend`, run:

```bash
python -m scripts.prepare_dataset
python -m scripts.train --dataset bbc
python -m scripts.evaluate --dataset bbc
```

Use `--dataset demo` to train on the included sample data. Training and evaluation outputs are saved under `backend/artifacts/`; trained models are saved under `backend/models/`.

The BBC texts are provided for educational use; their copyright belongs to the BBC. The dataset copy used for the reported experiments came from a third-party GitHub repository, and its identity with the original Kaggle dataset was not verified by checksum.

## Tests

```powershell
cd backend
pip install -r requirements-dev.txt
pytest
cd ..\frontend
npm ci
npm run build
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service and model status |
| `POST /api/summarize` | Summarize text |
| `POST /api/summarize/file` | Summarize a TXT, PDF, or DOCX upload |
| `GET /api/models` | Available summarization algorithms |
| `GET /api/dataset` | Dataset statistics |
| `GET /api/evaluation` | Evaluation results |
| `POST /api/train` | Start background model training |
| `GET /api/train/status` | Check training progress |

Summarization requests accept `text`, `language` (`auto`, `en`, or `uk`), `algorithm` (`random_forest`, `tfidf`, or `lead`), and either `num_sentences` or `ratio`.

## Method

Text is cleaned and split into sentences, then represented with TF-IDF and sentence-level features such as position, length, keyword frequency, and centrality. A Random Forest ranks sentences; the selected sentences are returned in their original order. If no trained model is available, the app uses a TF-IDF and centrality fallback.

Evaluation uses ROUGE-1, ROUGE-2, and ROUGE-L on a held-out test split. Results and charts are available in the app when the evaluation artifacts are present.
