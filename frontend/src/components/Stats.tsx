import type { SummarizeResponse } from "../types";

interface Props { result: SummarizeResponse; originalChars: number }

const fmt = (n: number) => n.toLocaleString("en-US");

export default function Stats({ result, originalChars }: Props) {
  const summaryChars = result.summary.length;
  const cut = (1 - result.compression_ratio) * 100;
  return (
    <div className="stats">
      <div className="card"><h4>Original text</h4>
        <dl>
          <dt>Words</dt><dd>{fmt(result.original_word_count)}</dd>
          <dt>Sentences</dt><dd>{fmt(result.original_sentence_count)}</dd>
          <dt>Characters</dt><dd>{fmt(originalChars)}</dd>
        </dl>
      </div>
      <div className="card"><h4>Summary</h4>
        <dl>
          <dt>Words</dt><dd>{fmt(result.summary_word_count)}</dd>
          <dt>Sentences</dt><dd>{fmt(result.summary_sentence_count)}</dd>
          <dt>Characters</dt><dd>{fmt(summaryChars)}</dd>
        </dl>
      </div>
      <div className="card"><h4>Metrics</h4>
        <dl>
          <dt>Compression ratio</dt><dd>{result.compression_ratio.toFixed(3)}</dd>
          <dt>Reduction</dt><dd>{cut.toFixed(1)}%</dd>
          <dt>Processing time</dt><dd>{(result.processing_time * 1000).toFixed(0)} ms</dd>
          <dt>Model</dt><dd>{result.model_name}</dd>
          <dt>Language</dt><dd>{result.language}</dd>
        </dl>
      </div>
    </div>
  );
}
