import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { errorMessage, getEvaluation } from "../api";
import type { EvaluationResult, MethodMetrics } from "../types";

const COLORS: Record<string, string> = { random: "#aeb7ad", lead: "#7d897f", tfidf: "#91aaa2", tfidf_centrality: "#c38d68", random_forest: "#687d54" };
const NAMES: Record<string, string> = { random: "Random", lead: "Lead-N", tfidf: "TF-IDF", tfidf_centrality: "TF-IDF + centrality", random_forest: "Random Forest" };
const f = (x: number, d = 3) => x.toFixed(d);

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<"f1" | "precision" | "recall">("f1");

  useEffect(() => { getEvaluation().then(setData).catch((e) => setError(errorMessage(e))); }, []);

  if (error) return <div className="error" role="alert">{error}</div>;
  if (!data) return <div className="loader" role="status" aria-label="Loading evaluation" />;

  const keys = Object.keys(data.methods);
  const rows: [string, MethodMetrics][] = keys.map((k) => [k, data.methods[k]]);
  const chart = ["rouge1", "rouge2", "rougeL"].map((r) => ({
    name: r === "rougeL" ? "ROUGE-L" : r === "rouge1" ? "ROUGE-1" : "ROUGE-2",
    ...Object.fromEntries(keys.map((k) => [NAMES[k], (data.methods[k] as any)[`${r}_${metric}`]])),
  }));
  const sweep = data.ratio_sweep.map((e) => ({ ratio: e.ratio, ...Object.fromEntries(keys.map((k) => [NAMES[k], e[k].rougeL_f1])) }));
  const rf = data.methods.random_forest;
  const bestBase = keys.filter((k) => k !== "random_forest").reduce((a, b) => (data.methods[a].rougeL_f1 > data.methods[b].rougeL_f1 ? a : b));
  const bs = data.paired_bootstrap[`random_forest_vs_${bestBase}`]?.rougeL;

  return (
    <div className="eval">
      <section className="panel">
        <p className="eyebrow">02 / BENCHMARK</p>
        <h2>Model comparison</h2>
        <p className="muted">Test set: {data.test_documents} documents ({data.dataset}); summary length: {f(data.ratio * 100, 1)}% of sentences; generated {data.generated_at}.</p>
        <div className="summary">
          Random Forest: ROUGE-1 F1 = {f(rf.rouge1_f1)}, ROUGE-2 F1 = {f(rf.rouge2_f1)}, ROUGE-L F1 = {f(rf.rougeL_f1)}.
          {" "}Best ROUGE-L baseline: {NAMES[bestBase]} ({f(data.methods[bestBase].rougeL_f1)}).
          {bs && <> ROUGE-L F1 difference (RF − {NAMES[bestBase]}) = {f(bs.mean_diff)}, 95% CI [{f(bs.ci95[0])}; {f(bs.ci95[1])}] ({bs.significant ? "statistically significant" : "not significant"}; RF wins on {bs.wins} of {bs.wins + bs.losses + bs.ties} documents).</>}
          {" "}BBC News Summary references are extractive, so absolute scores are higher than on abstractive datasets.
        </div>
      </section>

      <section className="panel">
        <h3>Metric scores <span className="muted small">MEAN OVER TEST SET</span></h3>
        <div className="tablewrap">
          <table>
            <thead>
              <tr><th rowSpan={2}>Model</th><th colSpan={3}>ROUGE-1</th><th colSpan={3}>ROUGE-2</th><th colSpan={3}>ROUGE-L</th><th rowSpan={2}>Compression</th><th rowSpan={2}>Time, ms</th></tr>
              <tr>{[1, 2, 3].map((i) => ["P", "R", "F1"].map((m) => <th key={`${i}${m}`}>{m}</th>))}</tr>
            </thead>
            <tbody>
              {rows.map(([k, m]) => (
                <tr key={k} className={k === "random_forest" ? "hl" : ""}>
                  <td>{NAMES[k]}</td>
                  <td>{f(m.rouge1_precision)}</td><td>{f(m.rouge1_recall)}</td><td><b>{f(m.rouge1_f1)}</b></td>
                  <td>{f(m.rouge2_precision)}</td><td>{f(m.rouge2_recall)}</td><td><b>{f(m.rouge2_f1)}</b></td>
                  <td>{f(m.rougeL_precision)}</td><td>{f(m.rougeL_recall)}</td><td><b>{f(m.rougeL_f1)}</b></td>
                  <td>{f(m.compression_words)}</td><td>{f(m.time_ms, 1)}</td>
                </tr>))}
            </tbody>
          </table>
        </div>
        <p className="muted small">Compression = summary words / original words. Time is the full generation time for one document.</p>
      </section>

      <section className="panel">
        <h3>ROUGE scores
          <select value={metric} onChange={(e) => setMetric(e.target.value as any)}>
            <option value="f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option>
          </select>
        </h3>
        <div className="chart">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chart}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
              {keys.map((k) => <Bar key={k} dataKey={NAMES[k]} fill={COLORS[k]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h3>ROUGE-L F1 by summary length</h3>
        <div className="chart">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={sweep}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="ratio" type="number" domain={[0.2, 0.5]} ticks={sweep.map((e) => e.ratio)} /><YAxis domain={[0.2, 0.6]} /><Tooltip /><Legend />
              {keys.map((k) => <Line key={k} type="monotone" dataKey={NAMES[k]} stroke={COLORS[k]} strokeWidth={2} />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {data.equal_word_budget && (
        <section className="panel">
          <h3>Equal word budget ({f(data.equal_word_budget.word_ratio * 100, 1)}% of original words)</h3>
          <table>
            <thead><tr><th>Model</th><th>ROUGE-1 F1</th><th>ROUGE-2 F1</th><th>ROUGE-L F1</th><th>Compression</th></tr></thead>
            <tbody>
              {Object.entries(data.equal_word_budget.methods).map(([k, m]) => (
                <tr key={k} className={k === "random_forest" ? "hl" : ""}><td>{NAMES[k]}</td><td>{f(m.rouge1_f1)}</td><td>{f(m.rouge2_f1)}</td><td>{f(m.rougeL_f1)}</td><td>{f(m.compression_words)}</td></tr>))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
