import axios from "axios";
import type { AlgorithmInfo, DatasetInfo, EvaluationResult, Health, Language, SummarizeResponse } from "./types";

const http = axios.create({ baseURL: import.meta.env.VITE_API_URL ?? "/api", timeout: 120_000 });

/** Converts Axios errors into readable messages. */
export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) {
    if (e.response) {
      const d = e.response.data as { detail?: unknown } | undefined;
      if (typeof d?.detail === "string") return d.detail;
      if (d?.detail) return JSON.stringify(d.detail);
      return `Server error (${e.response.status})`;
    }
    if (e.code === "ECONNABORTED") return "The request timed out.";
    return "Unable to reach the server. Check that the backend is running (http://localhost:8000).";
  }
  return e instanceof Error ? e.message : "An unknown error occurred.";
}

export interface SummarizeParams {
  language: Language;
  algorithm: string;
  numSentences?: number;
  ratio?: number; // fraction of original sentences to keep
}

export async function summarizeText(text: string, p: SummarizeParams): Promise<SummarizeResponse> {
  const { data } = await http.post<SummarizeResponse>("/summarize", {
    text, language: p.language, algorithm: p.algorithm,
    num_sentences: p.numSentences ?? null, ratio: p.ratio ?? null,
  });
  return data;
}

export async function summarizeFile(file: File, p: SummarizeParams): Promise<SummarizeResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", p.language);
  form.append("algorithm", p.algorithm);
  if (p.numSentences !== undefined) form.append("num_sentences", String(p.numSentences));
  if (p.ratio !== undefined) form.append("ratio", String(p.ratio));
  const { data } = await http.post<SummarizeResponse>("/summarize/file", form);
  return data;
}

export const getModels = async () => (await http.get<AlgorithmInfo[]>("/models")).data;
export const getHealth = async () => (await http.get<Health>("/health")).data;
export const getEvaluation = async () => (await http.get<EvaluationResult>("/evaluation")).data;
export const getDataset = async () => (await http.get<DatasetInfo>("/dataset")).data;
