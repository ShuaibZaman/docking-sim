import type { Checkpoint, Metrics, Replay, RunSummary } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function fetchRuns(): Promise<RunSummary[]> {
  return request("/api/runs");
}

export function fetchCheckpoints(runId: string): Promise<Checkpoint[]> {
  return request(`/api/runs/${encodeURIComponent(runId)}/checkpoints`);
}

export function fetchMetrics(runId: string): Promise<Metrics> {
  return request(`/api/runs/${encodeURIComponent(runId)}/metrics`);
}

export function fetchReplay(body: {
  run_id?: string | null;
  checkpoint: string;
  seed: number;
}): Promise<Replay> {
  return request("/api/replay", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
