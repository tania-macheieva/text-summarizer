import { useEffect, useRef, useState } from "react";
import { errorMessage, getModels, summarizeFile, summarizeText } from "../api";
import Stats from "../components/Stats";
import type { AlgorithmInfo, Language, LengthMode, SummarizeResponse } from "../types";

const ACCEPT = ".txt,.pdf,.docx";

export default function SummarizePage() {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<Language>("auto");
  const [mode, setMode] = useState<LengthMode>("percent");
  const [numSentences, setNumSentences] = useState(3);
  const [reduction, setReduction] = useState(60); // percentage reduction
  const [algorithm, setAlgorithm] = useState("random_forest");
  const [models, setModels] = useState<AlgorithmInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SummarizeResponse | null>(null);
  const [originalChars, setOriginalChars] = useState(0);
  const [copied, setCopied] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => { getModels().then(setModels).catch(() => setModels([])); }, []);
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const params = () => ({
    language, algorithm,
    ...(mode === "sentences" ? { numSentences } : { ratio: Math.max(0.01, (100 - reduction) / 100) }),
  });

  async function onSummarize() {
    setError(null); setCopied(false);
    if (!file && !text.trim()) { setError("Enter some text or upload a file."); return; }
    setLoading(true);
    try {
      const res = file ? await summarizeFile(file, params()) : await summarizeText(text, params());
      setResult(res);
      setOriginalChars(file ? 0 : text.length);
    } catch (e) {
      setResult(null); setError(errorMessage(e));
    } finally { setLoading(false); }
  }

  async function copy() {
    if (!result) return;
    await navigator.clipboard.writeText(result.summary);
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  }

  function download() {
    if (!result) return;
    const blob = new Blob([result.summary], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "summary.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <>
      <header className="page-heading">
        <div>
          <p className="eyebrow">01 / WORKSPACE</p>
          <h1>Find what matters.</h1>
        </div>
        <p>Paste a document or add a file. We’ll keep what matters and leave the noise behind.</p>
      </header>
      <div className="summarize-grid">
      <section className="panel input-panel">
        <div className="panel-head">
          <h2>Source text</h2>
          <span className="section-index">SOURCE / 01</span>
        </div>
        <label className="label source-label" htmlFor="source-text">DOCUMENT TEXT</label>
        <textarea
          id="source-text"
          className="text-input"
          value={text} onChange={(e) => { setText(e.target.value); if (file) setFile(null); }}
          placeholder="Paste an article, report, or other document…" rows={11} disabled={!!file}
        />
        <div className="text-meter">
          <span>{file ? "Text will be extracted from the file" : `${wordCount.toLocaleString("en-US")} words`}</span>
          <span>UA / EN</span>
        </div>
        <div className="upload-row">
          <input ref={fileInput} type="file" accept={ACCEPT} hidden
                 onChange={(e) => { setFile(e.target.files?.[0] ?? null); if (e.target.files?.[0]) setText(""); }} />
          <button className="upload-trigger" onClick={() => fileInput.current?.click()}>＋ Add a file · TXT / PDF / DOCX</button>
          {file && <span className="filetag"><span>{file.name}</span><button className="link" aria-label="Remove file" onClick={() => { setFile(null); if (fileInput.current) fileInput.current.value = ""; }}>×</button></span>}
        </div>

        <div className="options">
          <label>Language
            <select value={language} onChange={(e) => setLanguage(e.target.value as Language)}>
              <option value="auto">Detect automatically</option><option value="en">English</option><option value="uk">Ukrainian</option>
            </select>
          </label>
          <label>Algorithm
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {(models.length ? models : [{ id: "random_forest", name: "Random Forest" } as AlgorithmInfo]).map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>))}
            </select>
          </label>
          <fieldset>
            <legend>Summary length</legend>
            <div className="mode-choice">
              <label className="inline"><input type="radio" checked={mode === "percent"} onChange={() => setMode("percent")} /> Reduction</label>
              <label className="inline"><input type="radio" checked={mode === "sentences"} onChange={() => setMode("sentences")} /> Sentence count</label>
            </div>
            {mode === "percent" ? (
              <div className="slider"><input type="range" min={10} max={90} step={5} value={reduction} onChange={(e) => setReduction(+e.target.value)} />
                <span>Reduce by {reduction}% (keep ≈{100 - reduction}% of sentences)</span></div>
            ) : (
              <div className="slider"><input type="number" min={1} max={50} value={numSentences} onChange={(e) => setNumSentences(Math.max(1, +e.target.value || 1))} />
                <span>sentences in summary</span></div>
            )}
          </fieldset>
        </div>
        <button className="primary" onClick={onSummarize} disabled={loading}>{loading ? "Working on your text…" : "Create summary"}</button>
        {loading && <div className="loader" role="status" aria-label="loading" />}
        {error && <div className="error" role="alert">{error}</div>}
      </section>

      <section className="panel output-panel">
        <div className="panel-head">
          <h2>Summary</h2>
          <span className="section-index">OUTPUT / 02</span>
        </div>
        {!result && !loading && <div className="empty-result">
          <span className="empty-mark" aria-hidden="true">01</span>
          <p className="empty-caption"><strong>Room for the essentials.</strong>Your concise summary and the sentences behind it will appear here.</p>
        </div>}
        {result && (
          <>
            {result.warnings.map((w) => <div className="warn" key={w}>{w}</div>)}
            <p className="result-meta">{result.original_word_count.toLocaleString("en-US")} words → {result.summary_word_count.toLocaleString("en-US")} · {result.language.toUpperCase()} · {result.model_name}</p>
            <div className="summary" data-testid="summary">{result.summary}</div>
            <div className="result-actions">
              <button className="secondary" onClick={copy}>{copied ? "Copied ✓" : "Copy"}</button>
              <button className="secondary" onClick={download}>Download .txt</button>
            </div>
            <h3>Key sentences</h3>
            <ol className="keys">
              {result.key_sentences.map((k) => (
                <li key={k.index}>
                  <span className="badge">{String(k.rank).padStart(2, "0")}</span>
                  <span>{k.text}</span>
                  <span className="key-meta">SENTENCE {k.index + 1} · IMPORTANCE {k.score.toFixed(2)}</span>
                </li>))}
            </ol>
            <Stats result={result} originalChars={originalChars} />
            {result.filename && <p className="muted small file-note">Source: {result.filename}</p>}
          </>
        )}
      </section>
      </div>
    </>
  );
}
