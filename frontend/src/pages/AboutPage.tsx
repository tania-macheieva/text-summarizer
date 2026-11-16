import { useEffect, useState } from "react";
import { getDataset, getHealth, getModels } from "../api";
import type { AlgorithmInfo, DatasetInfo, Health } from "../types";

export default function AboutPage() {
  const [ds, setDs] = useState<DatasetInfo | null>(null);
  const [models, setModels] = useState<AlgorithmInfo[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    getDataset().then(setDs).catch(() => setDs(null));
    getModels().then(setModels).catch(() => setModels([]));
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);
  const st = ds?.statistics;
  return (
    <div className="about">
      <section className="panel">
        <h2>How it works</h2>
        <ol>
          <li>Text is cleaned and split into sentences (NLTK Punkt; Ukrainian uses a regex-based segmenter with abbreviation handling).</li>
          <li>Sentences are tokenized, stop words removed, words stemmed (Porter), and represented with TF-IDF.</li>
          <li>Each sentence is described by 18 features, including TF-IDF scores, position, length, keyword frequency, centrality (similarity and TextRank), and similarity to the document vector.</li>
          <li>A Random Forest classifier estimates the likelihood that each sentence belongs in a reference summary. Training labels are generated with a greedy ROUGE-L oracle.</li>
          <li>The top-ranked sentences are returned in their original order. If no trained model is available, the TF-IDF fallback is used.</li>
        </ol>
        {health && <p className="muted">Backend v{health.version}: model {health.model_loaded ? "loaded" : "not loaded (fallback)"}{health.model_trained_at ? `, trained ${health.model_trained_at}` : ""}.</p>}
      </section>
      <section className="panel">
        <h2>Dataset</h2>
        {!ds || !st ? <p className="muted">Dataset information is unavailable.</p> : (
          <>
            <p>{ds.name}: {st.documents} article-summary pairs across {Object.keys(st.categories).length} categories: {Object.entries(st.categories).map(([k, v]) => `${k} (${v})`).join(", ")}.</p>
            <ul>
              <li>Article length in words: mean {st.text_words.mean}, median {st.text_words.median}; summary length: mean {st.reference_words.mean}, median {st.reference_words.median}.</li>
              <li>Mean word compression ratio: {st.compression_words_mean}.</li>
              <li>Splits: {Object.entries(st.splits).map(([k, v]) => `${k} — ${v.documents}`).join(", ")} (seed {ds.seed}). {ds.leakage_check}.</li>
              <li>Exact duplicate articles: {st.exact_duplicate_texts}; documents in duplicate groups: {st.documents_in_duplicate_groups}.</li>
            </ul>
          </>)}
      </section>
      <section className="panel">
        <h2>Algorithms</h2>
        <ul>{models.map((m) => <li key={m.id}><b>{m.name}</b> — {m.description}</li>)}</ul>
      </section>
      <section className="panel">
        <h2>Technology</h2>
        <p>Backend: Python 3.12, FastAPI, Pydantic, scikit-learn, NLTK, NumPy, and Pandas. Evaluation: ROUGE (custom implementation, verified against rouge-score). Frontend: React, TypeScript, Vite, Axios, and Recharts. Docker Compose and pytest.</p>
      </section>
    </div>
  );
}
