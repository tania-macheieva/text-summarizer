export interface KeySentence { index: number; text: string; score: number; rank: number }

export interface SummarizeResponse {
  summary: string;
  key_sentences: KeySentence[];
  original_word_count: number;
  summary_word_count: number;
  compression_ratio: number;
  processing_time: number;
  model_name: string;
  language: string;
  original_sentence_count: number;
  summary_sentence_count: number;
  used_fallback: boolean;
  warnings: string[];
  filename?: string | null;
}

export interface AlgorithmInfo {
  id: string; name: string; description: string; available: boolean; trained: boolean;
  metadata: Record<string, unknown>;
}

export interface Health { status: string; version: string; model_loaded: boolean; model_trained_at?: string | null; uptime_seconds: number }

export type Language = "auto" | "en" | "uk";
export type LengthMode = "sentences" | "percent";

export interface MethodMetrics {
  label: string; documents: number;
  rouge1_precision: number; rouge1_recall: number; rouge1_f1: number;
  rouge2_precision: number; rouge2_recall: number; rouge2_f1: number;
  rougeL_precision: number; rougeL_recall: number; rougeL_f1: number;
  sent_precision: number; sent_recall: number; sent_f1: number;
  compression_words: number; time_ms: number;
}

export interface BootstrapResult { mean_diff: number; ci95: [number, number]; significant: boolean; wins: number; losses: number; ties: number }

export interface EvaluationResult {
  dataset: string;
  test_documents: number;
  ratio: number;
  methods: Record<string, MethodMetrics>;
  oracle: Partial<MethodMetrics> & { label: string };
  paired_bootstrap: Record<string, Record<string, BootstrapResult>>;
  equal_word_budget?: { word_ratio: number; methods: Record<string, MethodMetrics> } | null;
  ratio_sweep: Array<Record<string, any>>;
  model: { best_params: Record<string, unknown>; split_sizes: Record<string, number>; feature_importances: Array<{ feature: string; importance: number }>; trained_at: string; train_seconds: number };
  generated_at: string;
}

export interface DatasetInfo {
  name: string; path: string; seed: number; leakage_check: string;
  loading: Record<string, number>;
  validation: { input_rows: number; output_rows: number; total_dropped: number; dropped: Record<string, number> };
  statistics: {
    documents: number; categories: Record<string, number>;
    text_words: Record<string, number>; reference_words: Record<string, number>;
    text_sentences: Record<string, number>; reference_sentences: Record<string, number>;
    compression_words_mean: number; exact_duplicate_texts: number; documents_in_duplicate_groups: number;
    splits: Record<string, { documents: number }>;
  };
}
