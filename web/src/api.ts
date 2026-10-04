import type {
  Benchmark,
  BenchmarkCandidate,
  BenchmarkCompare,
  BenchmarkReplay,
  CandidateSelection,
  Checkpoint,
  EvalSummary,
  Metrics,
  Replay,
  RunSummary,
  ShowcasePreset,
} from "./types";

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

export function fetchEval(runId: string, benchmarkId?: string): Promise<EvalSummary> {
  const query = benchmarkId ? `?benchmark_id=${encodeURIComponent(benchmarkId)}` : "";
  return request(`/api/runs/${encodeURIComponent(runId)}/eval${query}`);
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

export function fetchBenchmarks(): Promise<Benchmark[]> {
  return request("/api/benchmarks");
}

export function fetchBenchmark(benchmarkId: string): Promise<Benchmark> {
  return request(`/api/benchmarks/${encodeURIComponent(benchmarkId)}`);
}

export function fetchBenchmarkCandidates(benchmarkId: string): Promise<BenchmarkCandidate[]> {
  return request(`/api/benchmarks/${encodeURIComponent(benchmarkId)}/candidates`);
}

export function fetchBenchmarkReplay(
  benchmarkId: string,
  body: {
    scenario_id: string;
    candidates: CandidateSelection[];
    max_steps?: number;
  },
): Promise<BenchmarkReplay> {
  return request(`/api/benchmarks/${encodeURIComponent(benchmarkId)}/replay`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchBenchmarkCompare(
  benchmarkId: string,
  candidates: CandidateSelection[],
): Promise<BenchmarkCompare> {
  return request(`/api/benchmarks/${encodeURIComponent(benchmarkId)}/compare`, {
    method: "POST",
    body: JSON.stringify({ candidates }),
  });
}

export function fetchPresets(): Promise<ShowcasePreset[]> {
  return request("/api/presets");
}

export function fetchBenchmarkExport(
  benchmarkId: string,
  body: {
    scenario_id: string;
    candidates: CandidateSelection[];
    max_steps?: number;
  },
): Promise<Record<string, unknown>> {
  return request(`/api/benchmarks/${encodeURIComponent(benchmarkId)}/export`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
