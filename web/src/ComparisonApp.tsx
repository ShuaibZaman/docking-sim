import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchBenchmark,
  fetchBenchmarkCandidates,
  fetchBenchmarkCompare,
  fetchBenchmarkExport,
  fetchBenchmarkReplay,
  fetchBenchmarks,
  fetchCheckpoints,
  fetchEval,
  fetchLevels,
  fetchPresets,
  fetchReplay,
  fetchRuns,
} from "./api";
import { SimCanvas } from "./SimCanvas";
import type {
  Benchmark,
  BenchmarkCandidate,
  BenchmarkCompare,
  BenchmarkPanel,
  CandidateSelection,
  Checkpoint,
  EvalOutcome,
  EvalPoint,
  EvalSummary,
  Frame,
  LevelChoice,
  PairwiseScenario,
  Replay,
  RunSummary,
  Selection,
  ShowcasePreset,
  World,
} from "./types";
import "./App.css";

const RANDOM_CHECKPOINT: Checkpoint = { id: "random", label: "Random policy", steps: 0, path: null };
const SPEEDS = [1, 2, 4, 8] as const;
const CURVE_COLORS = ["#3ee0c5", "#7eb6ff", "#e8a54b", "#e85d4c"];
type LabMode = "leaderboard" | "mission" | "explore";
type Pin = "" | "pd" | "random";
type GateState = "met" | "fail" | "neutral";

function checkpointLabel(checkpoint: Checkpoint): string {
  if (checkpoint.id === "random") return "Random";
  if (checkpoint.id === "pd") return "PD";
  if (checkpoint.id === "final") return "Final";
  if (checkpoint.steps <= 0) return checkpoint.label || checkpoint.id;
  return checkpoint.steps >= 1000 ? `${checkpoint.steps / 1000}k` : String(checkpoint.steps);
}

function runLabel(run: RunSummary): string {
  const family = (run.family || run.config_name || run.id).replace(/\.yaml$/i, "").replace(/_/g, " ");
  return `${family} · train ${run.seed}`;
}

function candidateKey(candidate: BenchmarkCandidate): string {
  return candidate.baseline ? `baseline:${candidate.baseline}` : `run:${candidate.id}`;
}

function selectionKey(selection: CandidateSelection): string {
  return selection.baseline ? `baseline:${selection.baseline}` : `run:${selection.run_id ?? ""}`;
}

function candidateTitle(candidate: BenchmarkCandidate): string {
  if (candidate.baseline) return candidate.label;
  const parts = [candidate.label.replace(/_/g, " ")];
  if (candidate.algorithm) parts.push(candidate.algorithm);
  if (candidate.architecture) parts.push(candidate.architecture);
  if (candidate.training_seed != null) parts.push(`seed ${candidate.training_seed}`);
  return parts.join(" · ");
}

function selectionFromCandidate(candidate: BenchmarkCandidate): CandidateSelection {
  if (candidate.baseline) {
    return { baseline: candidate.baseline, checkpoint: candidate.checkpoints[0]?.id ?? candidate.baseline };
  }
  return { run_id: candidate.id, checkpoint: "" };
}

function encodeSelection(selection: CandidateSelection): string {
  if (selection.baseline) return `b:${selection.baseline}`;
  return `r:${selection.run_id ?? ""}:${selection.checkpoint}`;
}

function decodeSelection(token: string): CandidateSelection | null {
  if (token.startsWith("b:")) {
    const baseline = token.slice(2);
    return baseline ? { baseline, checkpoint: baseline } : null;
  }
  if (!token.startsWith("r:")) return null;
  const rest = token.slice(2);
  const split = rest.indexOf(":");
  if (split <= 0) return null;
  return { run_id: rest.slice(0, split), checkpoint: rest.slice(split + 1) };
}

function readQuery(): { mode: LabMode; suite: string; scenario: string; pin: Pin; candidates: CandidateSelection[] } {
  const params = new URLSearchParams(window.location.search);
  const modeParam = params.get("mode");
  const mode: LabMode = modeParam === "explore" || modeParam === "leaderboard" || modeParam === "mission" ? modeParam : "mission";
  const pinParam = params.get("pin");
  const pin: Pin = pinParam === "pd" || pinParam === "random" ? pinParam : "";
  const candidates = (params.get("candidates") ?? "")
    .split(",")
    .map((token) => decodeSelection(token))
    .filter((item): item is CandidateSelection => item !== null)
    .slice(0, 4);
  return { mode, suite: params.get("suite") ?? "", scenario: params.get("scenario") ?? "", pin, candidates };
}

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function pairTone(row: PairwiseScenario): "success" | "crash" | "split" {
  if (row.left_success && row.right_success) return "success";
  if (!row.left_success && !row.right_success) return "crash";
  return "split";
}

function pairLabel(row: PairwiseScenario): string {
  if (row.left_success && row.right_success) return "both";
  if (row.left_success) return "left";
  if (row.right_success) return "right";
  return "neither";
}

function outcome(frame: Frame | undefined): { label: string; tone: "success" | "crash" | "neutral" } {
  if (!frame) return { label: "loading", tone: "neutral" };
  if (frame.success) return { label: "docked", tone: "success" };
  if (frame.terminal_reason) return { label: frame.terminal_reason.replace(/_/g, " "), tone: "crash" };
  if (frame.timeout) return { label: "timeout", tone: "neutral" };
  return { label: "in flight", tone: "neutral" };
}

function frameAt(panel: BenchmarkPanel, elapsedSeconds: number): Frame | undefined {
  const frames = panel.replay.frames;
  if (!frames.length) return undefined;
  const index = Math.min(frames.length - 1, Math.max(0, Math.floor(elapsedSeconds / panel.replay.dt)));
  return frames[index];
}

function insidePort(frame: Frame, world: World): boolean {
  const pose = frame.port_pose ?? { cx: world.port.cx, cy: world.port.cy, theta: 0, w: world.port.w, h: world.port.h };
  const dx = frame.x - pose.cx;
  const dy = frame.y - pose.cy;
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const lx = c * dx + s * dy;
  const ly = -s * dx + c * dy;
  return Math.abs(lx) <= pose.w / 2 && Math.abs(ly) <= pose.h / 2;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function formatMeasure(value: number | null | undefined, digits = 1, suffix = ""): string {
  if (value == null) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

function pointFor(summary: EvalSummary | undefined, checkpoint: string): EvalPoint | undefined {
  const points = summary?.points ?? [];
  return points.find((point) => point.checkpoint === checkpoint) ?? points[points.length - 1];
}

function orderedSelections(selections: CandidateSelection[], pin: Pin): CandidateSelection[] {
  if (!pin) return selections;
  return [...selections].sort((left, right) => Number(right.baseline === pin) - Number(left.baseline === pin));
}

type ScrubEvent = { time: number; kind: string; panel: string };

function scrubEvents(panels: BenchmarkPanel[]): ScrubEvent[] {
  const events: ScrubEvent[] = [];
  panels.forEach((panel) => {
    panel.replay.frames.forEach((frame) => {
      if (frame.t <= 0) return;
      const kind = frame.success
        ? "dock"
        : frame.hit_hull
          ? "hull"
          : frame.hit_asteroid
            ? "asteroid"
            : frame.out_of_bounds
              ? "bounds"
              : frame.out_of_fuel
                ? "fuel"
                : frame.timeout
                  ? "timeout"
                  : "";
      if (!kind || !(frame.done || frame.success)) return;
      events.push({ time: frame.t * panel.replay.dt, kind, panel: panel.label });
    });
  });
  return events;
}

function typingTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element) return false;
  const tag = element.tagName;
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || element.isContentEditable;
}

function captureReport(frame: Frame | undefined, world: World | undefined): { gates: Array<{ name: string; value: string; detail: string; state: GateState }>; sentence: string } {
  if (!frame || !world) return { gates: [], sentence: "Waiting for a frame." };
  const inPort = insidePort(frame, world);
  const headingDeg = (frame.heading_error * 180) / Math.PI;
  const spinLimit = world.dock_omega_max ?? 0.08;
  const gates = [
    {
      name: "In port",
      value: inPort ? "yes" : frame.distance.toFixed(2),
      detail: inPort ? "Inside the active port" : `${frame.distance.toFixed(2)} from the active port`,
      state: (inPort ? "met" : "fail") as GateState,
    },
    {
      name: "Speed",
      value: frame.speed.toFixed(2),
      detail: `speed ${frame.speed.toFixed(2)}, limit ${world.dock_speed_max}`,
      state: (frame.speed <= world.dock_speed_max ? "met" : "fail") as GateState,
    },
    {
      name: "Heading",
      value: `${headingDeg.toFixed(0)}°`,
      detail: `heading ${headingDeg.toFixed(1)}°, limit ${world.dock_angle_max_deg.toFixed(0)}°`,
      state: (headingDeg <= world.dock_angle_max_deg ? "met" : "fail") as GateState,
    },
    {
      name: "Spin",
      value: Math.abs(frame.omega).toFixed(2),
      detail: `spin ${Math.abs(frame.omega).toFixed(2)} rad/s, limit ${spinLimit}`,
      state: (Math.abs(frame.omega) <= spinLimit ? "met" : "fail") as GateState,
    },
  ];
  if (frame.success) return { gates, sentence: "Docked: stopped, facing the approach, and not spinning." };
  if (frame.phase === "hold") {
    const held = ((frame.hold ?? 0) * 0.05).toFixed(1);
    const need = (((frame.hold_steps ?? 20) * 0.05) || 1).toFixed(1);
    return { gates, sentence: `Holding at the first port (${held}s of ${need}s). Keep the stop.` };
  }
  if (frame.phase === "transfer" && !frame.done) {
    return { gates, sentence: "First port is done. Fly to the other port, stop, and face its approach." };
  }
  if (frame.done && !frame.success) return { gates, sentence: `Ended: ${frame.terminal_reason || "no dock"}.` };
  const failed = gates.filter((gate) => gate.state === "fail").map((gate) => gate.detail);
  return { gates, sentence: failed.length ? `Not docked: ${failed.join("; ")}.` : "Capture conditions are met." };
}

function CaptureGates({ frame, world }: { frame: Frame | undefined; world: World }) {
  const report = captureReport(frame, world);
  if (!report.gates.length) return null;
  return (
    <div className="gates" aria-label="Capture gates">
      {report.gates.map((gate) => (
        <span className={`gate ${gate.state}`} key={gate.name} title={gate.detail}>
          <em>{gate.name}</em>
          {gate.value}
        </span>
      ))}
    </div>
  );
}

function RewardBars({ frame }: { frame: Frame | undefined }) {
  const items = [
    ["Distance", frame?.components.distance ?? 0],
    ["Velocity", frame?.components.velocity ?? 0],
    ["Rotation", frame?.components.rotation ?? 0],
    ["Spin", frame?.components.spin ?? 0],
    ["Fuel", frame?.components.fuel ?? 0],
    ["Time", frame?.components.time ?? 0],
    ["Docking", frame?.components.terminal ?? 0],
  ] as const;
  const maxAbs = Math.max(1, ...items.map(([, value]) => Math.abs(value)));
  return (
    <div className="bars">
      {items.map(([label, value]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div className="bar-track">
            <div className={`bar-fill ${value >= 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(value) / maxAbs) * 100}%` }} />
          </div>
          <span>{value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function TelemetryGroups({ frame, frames }: { frame: Frame | undefined; frames: Frame[] }) {
  if (!frame) return null;
  const fuelUsed = frames[0] ? frames[0].fuel - frame.fuel : 0;
  const headingDeg = (frame.heading_error * 180) / Math.PI;
  const groups = [
    {
      title: "Trajectory",
      rows: [
        ["Position", `${frame.x.toFixed(2)}, ${frame.y.toFixed(2)}`],
        ["Relative v", `${(frame.relative_vx ?? frame.vx).toFixed(2)}, ${(frame.relative_vy ?? frame.vy).toFixed(2)}`],
      ],
    },
    {
      title: "Attitude",
      rows: [
        ["Heading err", `${headingDeg.toFixed(1)}°`],
        ["Yaw rate", frame.omega.toFixed(2)],
      ],
    },
    {
      title: "Propulsion",
      rows: [
        ["Axial / lat / yaw", `${frame.axial.toFixed(2)} / ${frame.lateral.toFixed(2)} / ${frame.yaw.toFixed(2)}`],
        ["Fuel used", fuelUsed.toFixed(1)],
      ],
    },
  ];
  return (
    <div className="telemetry">
      {groups.map((group) => (
        <section key={group.title}>
          <h4>{group.title}</h4>
          {group.rows.map(([label, value]) => (
            <p key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </p>
          ))}
        </section>
      ))}
    </div>
  );
}

function SuccessChart({ series }: { series: Array<{ name: string; points: Array<{ x: number; y: number }> }> }) {
  const drawable = series.filter((item) => item.points.length > 0);
  if (!drawable.length) return <p className="note">No stored checkpoint curve for the selected runs.</p>;
  const flat = drawable.flatMap((item) => item.points);
  const minX = Math.min(...flat.map((point) => point.x));
  const maxX = Math.max(...flat.map((point) => point.x));
  const xOf = (x: number) => (maxX === minX ? 160 : ((x - minX) / (maxX - minX)) * 300 + 8);
  const yOf = (y: number) => 108 - Math.max(0, Math.min(1, y)) * 92;
  return (
    <div className="curve-block">
      <svg className="chart curve-chart" viewBox="0 0 320 120" role="img" aria-label="Success versus training steps">
        {drawable.map((item, index) => (
          <path
            d={item.points.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${xOf(point.x).toFixed(1)},${yOf(point.y).toFixed(1)}`).join(" ")}
            fill="none"
            key={item.name}
            stroke={CURVE_COLORS[index % CURVE_COLORS.length]}
            strokeWidth="1.8"
          />
        ))}
      </svg>
      <ul className="chart-legend">
        {drawable.map((item, index) => (
          <li key={item.name}>
            <i style={{ background: CURVE_COLORS[index % CURVE_COLORS.length] }} />
            {item.name}
          </li>
        ))}
      </ul>
    </div>
  );
}

function StrataFailures({ point }: { point: EvalPoint | undefined }) {
  if (!point) return <p className="note">No stored benchmark evaluation for this checkpoint.</p>;
  const reasons = Object.entries(point.failure_reasons ?? {});
  return (
    <div className="strata-grid">
      <section>
        <h4>Strata</h4>
        {(point.strata ?? []).length ? (
          (point.strata ?? []).map((row) => (
            <div className="bar-row" key={row.stratum}>
              <span>{row.stratum.replace(/_/g, " ")}</span>
              <div className="bar-track">
                <div className="bar-fill pos" style={{ width: `${row.success_rate * 100}%` }} />
              </div>
              <span>{formatPercent(row.success_rate)}</span>
            </div>
          ))
        ) : (
          <p className="note">No stratum breakdown in this evaluation.</p>
        )}
      </section>
      <section>
        <h4>Terminal reasons</h4>
        {reasons.length ? (
          reasons.map(([reason, count]) => (
            <div className="bar-row" key={reason}>
              <span>{reason}</span>
              <div className="bar-track">
                <div className="bar-fill neg" style={{ width: `${(count / Math.max(point.n, 1)) * 100}%` }} />
              </div>
              <span>{count}</span>
            </div>
          ))
        ) : (
          <p className="note">No terminal-reason counts stored.</p>
        )}
      </section>
    </div>
  );
}

function EventScrubber({
  maxSeconds,
  value,
  events,
  onChange,
}: {
  maxSeconds: number;
  value: number;
  events: ScrubEvent[];
  onChange: (time: number) => void;
}) {
  const span = Math.max(maxSeconds, 0.05);
  return (
    <div className="scrubber">
      <div className="scrub-ticks">
        {events.map((event) => (
          <button
            className={`scrub-tick ${event.kind}`}
            key={`${event.panel}-${event.kind}-${event.time.toFixed(2)}`}
            onClick={() => onChange(event.time)}
            style={{ left: `${(event.time / span) * 100}%` }}
            title={`${event.panel} · ${event.kind} · ${event.time.toFixed(1)}s`}
            type="button"
          />
        ))}
      </div>
      <input
        className="slider scrub-range"
        max={maxSeconds}
        min={0}
        onChange={(event) => onChange(Number(event.target.value))}
        step={0.05}
        type="range"
        value={Math.min(value, maxSeconds)}
      />
    </div>
  );
}

function CandidateReplayCard({
  panel,
  title,
  checkpoint,
  elapsedSeconds,
  focused,
}: {
  panel: BenchmarkPanel;
  title: string;
  checkpoint: string;
  elapsedSeconds: number;
  focused: boolean;
}) {
  const [selection, setSelection] = useState<Selection | null>(null);
  const replay = panel.replay;
  const frames = replay.frames;
  const index = Math.min(frames.length - 1, Math.max(0, Math.floor(elapsedSeconds / replay.dt)));
  const frame = frames[index];
  const state = outcome(frame);
  return (
    <article className={focused ? "candidate-card focused" : "candidate-card"}>
      <header className="candidate-header">
        <div>
          <span className="candidate-label">{title}</span>
          <span className="candidate-checkpoint">{checkpoint}</span>
        </div>
        <span className={`outcome-inline ${state.tone}`}>{state.label}</span>
      </header>
      <div className="candidate-canvas">
        <SimCanvas world={replay.world} frames={frames} index={index} selection={selection} onSelect={setSelection} />
      </div>
    </article>
  );
}

function CandidateEditor({
  selection,
  candidates,
  onChange,
  onRemove,
}: {
  selection: CandidateSelection;
  candidates: BenchmarkCandidate[];
  onChange: (next: CandidateSelection) => void;
  onRemove: () => void;
}) {
  const compatible = candidates.filter((candidate) => candidate.compatible);
  const choice = compatible.find((candidate) => candidateKey(candidate) === selectionKey(selection));
  return (
    <div className="candidate-editor">
      <label>
        Candidate
        <select
          value={choice ? candidateKey(choice) : ""}
          onChange={(event) => {
            const next = compatible.find((candidate) => candidateKey(candidate) === event.target.value);
            if (next) onChange(selectionFromCandidate(next));
          }}
        >
          <option value="">choose a compatible run</option>
          {compatible.map((candidate) => (
            <option key={candidateKey(candidate)} value={candidateKey(candidate)}>
              {candidateTitle(candidate)}
            </option>
          ))}
        </select>
      </label>
      <label>
        Checkpoint
        <select
          value={selection.checkpoint}
          disabled={!choice || Boolean(choice.baseline)}
          onChange={(event) => onChange({ ...selection, checkpoint: event.target.value })}
        >
          <option value="">choose checkpoint</option>
          {(choice?.checkpoints ?? []).map((checkpoint) => (
            <option key={checkpoint.id} value={checkpoint.id}>
              {checkpointLabel(checkpoint)}
            </option>
          ))}
        </select>
      </label>
      <button className="quiet-button" onClick={onRemove} type="button">
        Remove
      </button>
    </div>
  );
}

function Explorer({ runs, active }: { runs: RunSummary[]; active: boolean }) {
  const [runId, setRunId] = useState("");
  const [checkpoint, setCheckpoint] = useState("random");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([RANDOM_CHECKPOINT]);
  const [levels, setLevels] = useState<LevelChoice[]>([]);
  const [level, setLevel] = useState(1);
  const [seed, setSeed] = useState(42);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(2);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const seededRun = useRef(false);

  useEffect(() => {
    fetchLevels().then(setLevels).catch(() => setLevels([]));
  }, []);

  useEffect(() => {
    if (seededRun.current || runId || !runs[0]) return;
    seededRun.current = true;
    setRunId(runs[0].id);
  }, [runId, runs]);

  useEffect(() => {
    if (!runId) {
      setCheckpoints([RANDOM_CHECKPOINT]);
      setCheckpoint("random");
      return;
    }
    fetchCheckpoints(runId)
      .then((items) => {
        setCheckpoints(items);
        setCheckpoint((current) => (items.some((item) => item.id === current) ? current : items[0]?.id ?? "random"));
      })
      .catch((err: Error) => setError(err.message));
  }, [runId]);

  useEffect(() => {
    let cancelled = false;
    setPlaying(false);
    setIndex(0);
    setSelection(null);
    setLoading(true);
    fetchReplay({ run_id: runId || null, checkpoint, seed, level })
      .then((next) => {
        if (!cancelled) {
          setReplay(next);
          setError(null);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, checkpoint, seed, level]);

  const frames = replay?.frames ?? [];
  useEffect(() => {
    if (!playing || frames.length < 2 || !replay) return;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= frames.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, (replay.dt * 1000) / speed);
    return () => window.clearInterval(timer);
  }, [playing, frames.length, replay, speed]);

  useEffect(() => {
    if (!active) return;
    const onKey = (event: KeyboardEvent) => {
      if (typingTarget(event.target)) return;
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((value) => !value);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setPlaying(false);
        setIndex((current) => Math.min(frames.length - 1, current + 1));
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setPlaying(false);
        setIndex((current) => Math.max(0, current - 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, frames.length]);

  const current = frames[index];
  const state = outcome(current);
  const elapsed = replay ? index * replay.dt : 0;
  const report = captureReport(current, replay?.world);
  const selectedRun = runs.find((run) => run.id === runId);
  const meta = replay?.meta;
  const levelSummary = levels.find((item) => item.id === level)?.summary;
  return (
    <section className="explore-shell compact-explore">
      <div className="header-controls explore-toolbar">
        <label>
          Run
          <select value={runId} onChange={(event) => setRunId(event.target.value)}>
            <option value="">Random policy</option>
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {runLabel(run)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Checkpoint
          <select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)}>
            {checkpoints.map((item) => (
              <option key={item.id} value={item.id}>
                {checkpointLabel(item)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Level
          <select value={level} onChange={(event) => setLevel(Number(event.target.value))}>
            {levels.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}. {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Seed
          <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        </label>
      </div>
      <div className="main">
        <section className="sim-pane">
          {loading || !replay ? (
            <div className="skeleton" />
          ) : (
            <SimCanvas world={replay.world} frames={frames} index={index} selection={selection} onSelect={setSelection} />
          )}
          <span className={`outcome ${state.tone}`}>{state.label}</span>
          <div className="explore-footer canvas-footer">
            <button onClick={() => setPlaying((value) => !value)} type="button">
              {playing ? "Pause" : "Play"}
            </button>
            <label>
              Speed
              <select value={speed} onChange={(event) => setSpeed(Number(event.target.value) as (typeof SPEEDS)[number])}>
                {SPEEDS.map((value) => (
                  <option key={value} value={value}>
                    {value}x
                  </option>
                ))}
              </select>
            </label>
            <input
              className="slider"
              type="range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={index}
              onChange={(event) => {
                setPlaying(false);
                setIndex(Number(event.target.value));
              }}
            />
            <span>
              {elapsed.toFixed(1)}s · {index}/{Math.max(0, frames.length - 1)}
            </span>
          </div>
        </section>
        <aside className="side">
          <h2>Model</h2>
          <div className="meta-grid">
            <div className="meta"><span className="k">Algorithm</span><span className="v">{meta?.algorithm ?? selectedRun?.algorithm ?? "Random"}</span></div>
            <div className="meta"><span className="k">Architecture</span><span className="v">{meta?.architecture ?? "—"}</span></div>
            <div className="meta"><span className="k">Parameters</span><span className="v">{meta?.n_params ?? selectedRun?.n_params ?? "—"}</span></div>
            <div className="meta"><span className="k">Optimizer</span><span className="v">{meta?.optimizer ?? "—"}</span></div>
            <div className="meta"><span className="k">Critic loss</span><span className="v">{meta?.critic_loss ?? "—"}</span></div>
            <div className="meta"><span className="k">Seed</span><span className="v">{selectedRun?.seed ?? seed}</span></div>
            <div className="meta"><span className="k">Observation</span><span className="v">{meta?.obs_mode ?? selectedRun?.obs_mode ?? "state"}</span></div>
            <div className="meta"><span className="k">Level</span><span className="v">{level}</span></div>
          </div>
          <h2>Why</h2>
          <p className="note">{report.sentence}</p>
          {levelSummary ? <p className="note">{levelSummary}</p> : null}
          {replay && current ? <CaptureGates frame={current} world={replay.world} /> : null}
          <h2>Reward</h2>
          <RewardBars frame={current} />
          <p className="note">Explore is a custom seed or a level preview. It is not benchmark evidence.</p>
          {error ? <p className="error">{error}</p> : null}
        </aside>
      </div>
    </section>
  );
}

function BenchmarkWorkspace({ mode, active }: { mode: LabMode; active: boolean }) {
  const initial = useRef(readQuery());
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [presets, setPresets] = useState<ShowcasePreset[]>([]);
  const [benchmarkId, setBenchmarkId] = useState(initial.current.suite);
  const [manifest, setManifest] = useState<Benchmark | null>(null);
  const [available, setAvailable] = useState<BenchmarkCandidate[]>([]);
  const [scenarioId, setScenarioId] = useState(initial.current.scenario);
  const [stratumFilter, setStratumFilter] = useState("all");
  const [selections, setSelections] = useState<CandidateSelection[]>(initial.current.candidates);
  const [pin, setPin] = useState<Pin>(initial.current.pin);
  const [panels, setPanels] = useState<BenchmarkPanel[]>([]);
  const [comparison, setComparison] = useState<BenchmarkCompare | null>(null);
  const [evals, setEvals] = useState<Record<string, EvalSummary>>({});
  const [focusRunId, setFocusRunId] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(2);
  const [highlight, setHighlight] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadedSuite = useRef<string | null>(null);

  useEffect(() => {
    const kind = initial.current.pin;
    if (!kind) return;
    setSelections((current) => {
      if (current.some((item) => item.baseline === kind)) return current;
      return [{ baseline: kind, checkpoint: kind }, ...current].slice(0, 4);
    });
  }, []);

  useEffect(() => {
    Promise.all([fetchBenchmarks(), fetchPresets()])
      .then(([items, nextPresets]) => {
        setBenchmarks(items);
        setPresets(nextPresets);
        setBenchmarkId((current) => current || items[0]?.id || "");
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!benchmarkId) return;
    const previousSuite = loadedSuite.current;
    const userChangedSuite = previousSuite !== null && previousSuite !== benchmarkId;
    loadedSuite.current = benchmarkId;
    Promise.all([fetchBenchmark(benchmarkId), fetchBenchmarkCandidates(benchmarkId)])
      .then(([nextManifest, nextCandidates]) => {
        setManifest(nextManifest);
        setAvailable(nextCandidates);
        setScenarioId((current) => {
          const scenarios = nextManifest.scenarios ?? [];
          if (!userChangedSuite && current && scenarios.some((scenario) => scenario.id === current)) return current;
          return scenarios[0]?.id ?? "";
        });
        if (userChangedSuite) {
          setSelections([]);
          setPanels([]);
          setComparison(null);
          setElapsedSeconds(0);
          setPlaying(false);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, [benchmarkId]);

  useEffect(() => {
    if (!benchmarkId) return;
    const ids = available.filter((candidate) => candidate.compatible && !candidate.baseline).map((candidate) => candidate.id);
    let cancelled = false;
    Promise.all(
      ids.map(async (id) => {
        try {
          return [id, await fetchEval(id, benchmarkId)] as const;
        } catch {
          return [id, null] as const;
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      const next: Record<string, EvalSummary> = {};
      entries.forEach(([id, summary]) => {
        if (summary) next[id] = summary;
      });
      setEvals(next);
      setFocusRunId((current) => current || ids.find((id) => next[id]?.points?.length) || ids[0] || "");
    });
    return () => {
      cancelled = true;
    };
  }, [available, benchmarkId]);

  useEffect(() => {
    const params = new URLSearchParams();
    params.set("mode", mode);
    if (benchmarkId) params.set("suite", benchmarkId);
    if (scenarioId) params.set("scenario", scenarioId);
    if (pin) params.set("pin", pin);
    if (selections.length) params.set("candidates", selections.map(encodeSelection).join(","));
    const next = `${window.location.pathname}?${params.toString()}`;
    if (`${window.location.pathname}${window.location.search}` !== next) {
      window.history.replaceState(null, "", next);
    }
  }, [benchmarkId, mode, pin, scenarioId, selections]);

  const scenarios = manifest?.scenarios ?? [];
  const selectedScenario = scenarios.find((scenario) => scenario.id === scenarioId);
  const visibleScenarios = stratumFilter === "all" ? scenarios : scenarios.filter((scenario) => scenario.stratum === stratumFilter);
  const missionPosition = Math.max(0, scenarios.findIndex((scenario) => scenario.id === scenarioId));
  const validSelections = orderedSelections(
    selections.filter((selection) => Boolean(selection.checkpoint) && (selection.run_id || selection.baseline)),
    pin,
  );
  const trainedSelections = validSelections.filter((selection) => selection.run_id && !selection.baseline);
  const maxSeconds = Math.max(0, ...panels.map((panel) => panel.replay.steps * panel.replay.dt));
  const incompatible = available.filter((candidate) => !candidate.compatible);
  const primaryRun = trainedSelections[0]?.run_id ?? focusRunId;
  const primaryCheckpoint = trainedSelections[0]?.checkpoint ?? pointFor(evals[primaryRun], "")?.checkpoint ?? "";
  const primaryPoint = pointFor(evals[primaryRun], primaryCheckpoint);
  const primaryOutcomes = new Map<string, EvalOutcome>(
    (evals[primaryRun]?.latest_outcomes ?? [])
      .filter((row) => row.scenario_id)
      .map((row) => [String(row.scenario_id), row]),
  );

  useEffect(() => {
    if (!playing || !panels.length) return;
    const timer = window.setInterval(() => {
      setElapsedSeconds((time) => {
        if (time >= maxSeconds) {
          setPlaying(false);
          return maxSeconds;
        }
        return Math.min(maxSeconds, time + 0.05);
      });
    }, 50 / speed);
    return () => window.clearInterval(timer);
  }, [maxSeconds, panels.length, playing, speed]);

  useEffect(() => {
    if (!active || mode !== "mission") return;
    const onKey = (event: KeyboardEvent) => {
      if (typingTarget(event.target)) return;
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((value) => !value);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setPlaying(false);
        setElapsedSeconds((time) => Math.min(maxSeconds, time + 0.25));
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setPlaying(false);
        setElapsedSeconds((time) => Math.max(0, time - 0.25));
      } else if (event.key >= "1" && event.key <= "4") {
        setHighlight(Number(event.key) - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, maxSeconds, mode]);

  const liveSummary = useMemo(() => {
    if (!panels.length) return "awaiting mission";
    const frames = panels.map((panel) => frameAt(panel, elapsedSeconds));
    if (frames.some((frame) => frame && !frame.done)) return "in flight";
    const docked = frames.filter((frame) => frame?.success).length;
    return `${docked} of ${panels.length} docked`;
  }, [elapsedSeconds, panels]);

  const matchSelection = (selection: CandidateSelection) =>
    available.find((candidate) =>
      selection.baseline ? candidate.baseline === selection.baseline : candidate.id === selection.run_id,
    );

  const titleFor = (selection: CandidateSelection) => {
    const match = matchSelection(selection);
    return match ? candidateTitle(match) : selection.run_id || selection.baseline || "candidate";
  };

  const checkpointText = (selection: CandidateSelection) => {
    const checkpoint = matchSelection(selection)?.checkpoints.find((item) => item.id === selection.checkpoint);
    return checkpoint ? checkpointLabel(checkpoint) : selection.checkpoint;
  };

  const clearReplay = () => {
    setPanels([]);
    setComparison(null);
    setElapsedSeconds(0);
    setPlaying(false);
  };

  const updateSelection = (index: number, next: CandidateSelection) => {
    setSelections((current) => current.map((item, itemIndex) => (itemIndex === index ? next : item)));
    clearReplay();
  };

  const removeSelection = (index: number) => {
    setSelections((current) => current.filter((_item, itemIndex) => itemIndex !== index));
    clearReplay();
  };

  const togglePin = (kind: Pin) => {
    if (!kind) return;
    const nextPin: Pin = pin === kind ? "" : kind;
    setPin(nextPin);
    setSelections((current) => {
      const rest = current.filter((item) => item.baseline !== "pd" && item.baseline !== "random");
      if (!nextPin) return rest;
      return [{ baseline: nextPin, checkpoint: nextPin }, ...rest].slice(0, 4);
    });
    clearReplay();
  };

  const runMission = async (nextScenario = scenarioId) => {
    if (!benchmarkId || !nextScenario || !validSelections.length) return;
    setLoading(true);
    setError(null);
    setPlaying(false);
    try {
      const response = await fetchBenchmarkReplay(benchmarkId, {
        scenario_id: nextScenario,
        candidates: validSelections,
      });
      const ordered = orderedSelections(
        response.panels.map((panel) => panel.candidate),
        pin,
      );
      const rank = new Map(ordered.map((selection, index) => [selectionKey(selection), index]));
      setPanels(
        [...response.panels].sort(
          (left, right) => (rank.get(selectionKey(left.candidate)) ?? 0) - (rank.get(selectionKey(right.candidate)) ?? 0),
        ),
      );
      setElapsedSeconds(0);
      setHighlight(0);
      if (trainedSelections.length >= 2) {
        setComparison(await fetchBenchmarkCompare(benchmarkId, trainedSelections));
      } else {
        setComparison(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const openScenario = (id: string) => {
    setScenarioId(id);
    setPlaying(false);
    if (validSelections.length) void runMission(id);
    else clearReplay();
  };

  const curveSeries = trainedSelections.slice(0, 4).map((selection) => {
    const summary = selection.run_id ? evals[selection.run_id] : undefined;
    const candidate = available.find((item) => item.id === selection.run_id);
    return {
      name: candidate ? candidateTitle(candidate) : selection.run_id ?? "run",
      points: (summary?.points ?? [])
        .map((point) => ({ x: point.timesteps, y: point.success_rate }))
        .sort((left, right) => left.x - right.x),
    };
  });

  const boardRows = available
    .filter((candidate) => candidate.compatible && !candidate.baseline)
    .map((candidate) => ({ candidate, point: pointFor(evals[candidate.id], "") }))
    .sort((left, right) => (right.point?.success_rate ?? -1) - (left.point?.success_rate ?? -1));
  const bestSuccess = Math.max(...boardRows.map((row) => row.point?.success_rate ?? -1));
  const fuelValues = boardRows.map((row) => row.point?.mean_fuel_success).filter((value): value is number => value != null);
  const timeValues = boardRows.map((row) => row.point?.mean_time_success).filter((value): value is number => value != null);
  const crashValues = boardRows.map((row) => row.point?.crash_rate).filter((value): value is number => value != null);
  const bestFuel = fuelValues.length ? Math.min(...fuelValues) : null;
  const bestTime = timeValues.length ? Math.min(...timeValues) : null;
  const bestCrash = crashValues.length ? Math.min(...crashValues) : null;
  const focusPoint = pointFor(evals[focusRunId], "");
  const focusCurves = boardRows.slice(0, 4).map((row) => ({
    name: candidateTitle(row.candidate),
    points: (evals[row.candidate.id]?.points ?? [])
      .map((point) => ({ x: point.timesteps, y: point.success_rate }))
      .sort((left, right) => left.x - right.x),
  }));
  const pinLabel = pin === "pd" ? "PD rendezvous" : pin === "random" ? "Random policy" : "";
  const shownIndex = panels.length ? Math.min(highlight, panels.length - 1) : 0;
  const shownPanel = panels[shownIndex];
  const shownFrame = shownPanel ? frameAt(shownPanel, elapsedSeconds) : undefined;
  const shownReport = captureReport(shownFrame, shownPanel?.replay.world);

  const side = (
    <aside className="ops-side">
      <label>
        Suite
        <select value={benchmarkId} onChange={(event) => setBenchmarkId(event.target.value)}>
          {benchmarks.map((benchmark) => (
            <option key={benchmark.id} value={benchmark.id}>
              {benchmark.label} · {benchmark.n_scenarios}
            </option>
          ))}
        </select>
      </label>
      <div className="pin-row">
        <span>Baseline</span>
        <button className={pin === "pd" ? "tab active" : "tab"} onClick={() => togglePin("pd")} type="button">
          Pin PD
        </button>
        <button className={pin === "random" ? "tab active" : "tab"} onClick={() => togglePin("random")} type="button">
          Pin Random
        </button>
      </div>
      {mode === "mission" ? (
        <>
          <div className="preset-chips">
            {presets
              .filter((preset) => preset.benchmark_id === benchmarkId)
              .map((preset) => (
                <button
                  className={preset.scenario_id === scenarioId ? "scenario-chip active" : "scenario-chip"}
                  key={preset.id}
                  onClick={() => openScenario(preset.scenario_id)}
                  title={preset.description}
                  type="button"
                >
                  {preset.label}
                </button>
              ))}
          </div>
          <div className="mission-stepper">
            <button disabled={missionPosition <= 0} onClick={() => openScenario(scenarios[missionPosition - 1].id)} type="button">
              Previous
            </button>
            <div>
              <span className="eyebrow">Mission {scenarios.length ? missionPosition + 1 : 0} of {scenarios.length}</span>
              <strong>{selectedScenario ? selectedScenario.stratum.replace(/_/g, " ") : "Choose a mission"}</strong>
            </div>
            <button
              disabled={missionPosition < 0 || missionPosition >= scenarios.length - 1}
              onClick={() => openScenario(scenarios[missionPosition + 1].id)}
              type="button"
            >
              Next
            </button>
          </div>
          <details className="incompatible-fold">
            <summary>All missions</summary>
            <label>
              Stratum
              <select value={stratumFilter} onChange={(event) => setStratumFilter(event.target.value)}>
                <option value="all">All strata</option>
                {(manifest?.strata ?? []).map((stratum) => (
                  <option key={stratum} value={stratum}>
                    {stratum.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
            {trainedSelections[0] && !pointFor(evals[trainedSelections[0].run_id ?? ""], trainedSelections[0].checkpoint) ? (
              <p className="note">No stored benchmark evaluation for the selected run.</p>
            ) : null}
            <div className="mission-list" aria-label="Benchmark missions">
              {visibleScenarios.map((scenario) => {
                const stored = primaryOutcomes.get(scenario.id);
                return (
                  <button
                    className={scenario.id === scenarioId ? "mission-row active" : "mission-row"}
                    key={scenario.id}
                    onClick={() => openScenario(scenario.id)}
                    type="button"
                  >
                    <span>{scenario.id.replace("static-v1-", "#")}</span>
                    <span>{scenario.stratum.replace(/_/g, " ")}</span>
                    <em className={stored?.success ? "good" : stored ? "bad" : ""}>
                      {stored ? (stored.success ? "docked" : stored.terminal_reason || "miss") : "—"}
                    </em>
                  </button>
                );
              })}
            </div>
          </details>
          <div className="section-heading">
            <h3>Candidates</h3>
            <button
              disabled={selections.length >= 4 || !available.some((candidate) => candidate.compatible)}
              onClick={() => setSelections((current) => [...current, { checkpoint: "" }])}
              type="button"
            >
              Add
            </button>
          </div>
          {selections.length === 0 ? (
            <p className="empty">Add a candidate and choose its checkpoint. Final is never substituted.</p>
          ) : (
            selections.map((selection, index) => (
              <CandidateEditor
                candidates={available}
                key={`${selectionKey(selection)}-${index}`}
                onChange={(next) => updateSelection(index, next)}
                onRemove={() => removeSelection(index)}
                selection={selection}
              />
            ))
          )}
          {incompatible.length ? (
            <details className="incompatible-fold">
              <summary>{incompatible.length} incompatible</summary>
              <ul>
                {incompatible.map((candidate) => (
                  <li key={candidate.id} title={candidate.reason ?? "incompatible"}>
                    {candidateTitle(candidate)}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
          <div className="action-row">
            <button className="primary-button" disabled={loading || validSelections.length === 0} onClick={() => void runMission()} type="button">
              {loading ? "Loading…" : "Run mission"}
            </button>
            <button
              disabled={loading || validSelections.length === 0 || !scenarioId}
              onClick={async () => {
                if (!benchmarkId || !scenarioId || !validSelections.length) return;
                try {
                  const bundle = await fetchBenchmarkExport(benchmarkId, {
                    scenario_id: scenarioId,
                    candidates: validSelections,
                  });
                  downloadJson(`${benchmarkId}-${scenarioId}.json`, bundle);
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err));
                }
              }}
              type="button"
            >
              Export JSON
            </button>
          </div>
        </>
      ) : (
        <p className="note">Leaderboard rows use stored benchmark scores for this suite. Replay a mission from the Mission tab.</p>
      )}
      {error ? <p className="error">{error}</p> : null}
    </aside>
  );

  const missionMain = (
    <div className="ops-main">
      <div className="playback-bar">
        <button onClick={() => setPlaying((value) => !value)} type="button" disabled={!panels.length}>
          {playing ? "Pause" : "Play"}
        </button>
        <label>
          Speed
          <select value={speed} onChange={(event) => setSpeed(Number(event.target.value) as (typeof SPEEDS)[number])}>
            {SPEEDS.map((value) => (
              <option key={value} value={value}>
                {value}x
              </option>
            ))}
          </select>
        </label>
        <EventScrubber
          events={scrubEvents(panels)}
          maxSeconds={maxSeconds}
          onChange={(time) => {
            setPlaying(false);
            setElapsedSeconds(time);
          }}
          value={elapsedSeconds}
        />
        <span className="hash">{elapsedSeconds.toFixed(1)}s</span>
      </div>
      <div className="ops-stage">
      {loading ? (
        <div className="panel-grid panels-2">
          <div className="skeleton" />
          <div className="skeleton" />
        </div>
      ) : panels.length ? (
        <section className={`panel-grid panels-${Math.min(panels.length, 4)}`}>
          {panels.map((panel, index) => (
            <CandidateReplayCard
              elapsedSeconds={elapsedSeconds}
              focused={shownIndex === index}
              key={`${panel.label}-${panel.candidate.checkpoint}-${index}`}
              panel={panel}
              checkpoint={checkpointText(panel.candidate)}
              title={titleFor(panel.candidate)}
            />
          ))}
        </section>
      ) : (
        <p className="empty viewport-empty">Run an exact mission to fill the viewports. Space plays, arrows scrub, and 1–4 highlight a panel.</p>
      )}
      {shownPanel && shownFrame ? (
        <section className="inspector">
          <p className="note">{shownReport.sentence}</p>
          <CaptureGates frame={shownFrame} world={shownPanel.replay.world} />
          <TelemetryGroups frame={shownFrame} frames={shownPanel.replay.frames} />
        </section>
      ) : null}
      <details className="evidence-fold">
        <summary>Evidence</summary>
      <section className="evidence">
        {pinLabel ? (
          <p className="note">
            {pinLabel} has no stored benchmark scores, so success, fuel, and time deltas against that baseline are not a ranking.
          </p>
        ) : null}
        {comparison?.pairwise.length ? (
          comparison.pairwise.map((pair) => (
            <article className="paired-summary" key={`${pair.left.run_id}-${pair.right.run_id}-${pair.left.checkpoint}-${pair.right.checkpoint}`}>
              <span className="eyebrow">
                {titleFor(pair.left)} {checkpointText(pair.left)} vs {titleFor(pair.right)} {checkpointText(pair.right)}
              </span>
              <div className="paired-grid">
                <span>Common <strong>{pair.summary.n_common}</strong></span>
                <span>Both <strong>{pair.summary.outcomes.both_success}</strong></span>
                <span>Left only <strong>{pair.summary.outcomes.left_only}</strong></span>
                <span>Right only <strong>{pair.summary.outcomes.right_only}</strong></span>
                <span>Neither <strong>{pair.summary.outcomes.neither}</strong></span>
                <span>
                  Fuel Δ<strong>{formatMeasure(pair.summary.mean_fuel_delta_left_minus_right, 2)}</strong>
                </span>
                <span>
                  Time Δ<strong>{formatMeasure(pair.summary.mean_time_delta_left_minus_right, 2, "s")}</strong>
                </span>
              </div>
              {pair.summary.n_common === 0 ? (
                <p className="note">No stored benchmark evaluation yet. This replay is exact, but it is not a ranking until the benchmark is scored.</p>
              ) : (
                <div className="scenario-chips">
                  {(pair.summary.scenarios ?? []).map((row) => (
                    <button
                      className={`scenario-chip pair-${pairTone(row)}${row.scenario_id === scenarioId ? " active" : ""}`}
                      key={row.scenario_id}
                      onClick={() => openScenario(row.scenario_id)}
                      type="button"
                    >
                      {row.scenario_id.replace("static-v1-", "#")} {pairLabel(row)}
                    </button>
                  ))}
                </div>
              )}
            </article>
          ))
        ) : null}
        <div className="evidence-split">
          <section>
            <h3>Learning curve</h3>
            <SuccessChart series={curveSeries} />
          </section>
          <section>
            <h3>Where it fails</h3>
            <StrataFailures point={primaryPoint} />
          </section>
        </div>
      </section>
      </details>
      </div>
    </div>
  );

  const leaderboardMain = (
    <div className="ops-main leaderboard-main">
      {pinLabel ? (
        <p className="note">
          {pinLabel} has no stored benchmark scores. Leaderboard colors compare trained runs with each other, not against that baseline.
        </p>
      ) : (
        <p className="note">Pin PD or Random when you want a baseline. Deltas appear only after that baseline has stored scores.</p>
      )}
      {boardRows.length === 0 ? (
        <p className="empty">No compatible trained runs for this suite.</p>
      ) : (
        <table className="matrix leaderboard">
          <thead>
            <tr>
              <th>Candidate</th>
              <th>Success</th>
              <th>Interval</th>
              <th>Crash</th>
              <th>Fuel on success</th>
              <th>Time on success</th>
              {pin ? <th>Δ vs {pin}</th> : null}
            </tr>
          </thead>
          <tbody>
            {boardRows.map((row) => {
              const point = row.point;
              const scored = Boolean(point);
              return (
                <tr
                  className={focusRunId === row.candidate.id ? "active-row" : ""}
                  key={row.candidate.id}
                  onClick={() => setFocusRunId(row.candidate.id)}
                >
                  <td>{candidateTitle(row.candidate)}</td>
                  <td className={point && point.success_rate === bestSuccess ? "good" : ""}>{scored ? formatPercent(point?.success_rate) : "no stored eval"}</td>
                  <td>{scored ? `${formatPercent(point?.success_ci_low)}–${formatPercent(point?.success_ci_high)}` : "—"}</td>
                  <td className={point && point.crash_rate === bestCrash ? "good" : ""}>{scored ? formatPercent(point?.crash_rate) : "—"}</td>
                  <td className={point?.mean_fuel_success != null && point.mean_fuel_success === bestFuel ? "good" : ""}>
                    {formatMeasure(point?.mean_fuel_success, 1)}
                  </td>
                  <td className={point?.mean_time_success != null && point.mean_time_success === bestTime ? "good" : ""}>
                    {formatMeasure(point?.mean_time_success, 1, "s")}
                  </td>
                  {pin ? <td>not scored</td> : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <div className="evidence-split">
        <section>
          <h3>Learning curves</h3>
          <SuccessChart series={focusCurves} />
        </section>
        <section>
          <h3>Where it fails</h3>
          <StrataFailures point={focusPoint} />
        </section>
      </div>
    </div>
  );

  const suiteLabel = benchmarks.find((item) => item.id === benchmarkId)?.label ?? (benchmarkId || "Suite");
  return (
    <div className="ops-shell">
      <div className="status-strip" aria-label="Lab status">
        <span>{suiteLabel}</span>
        <span>{selectedScenario?.id ?? "mission"}</span>
        <span>{selectedScenario ? selectedScenario.stratum.replace(/_/g, " ") : "—"}</span>
        <span>
          mission {scenarios.length ? missionPosition + 1 : 0} of {scenarios.length}
        </span>
        {mode === "mission" ? (
          <>
            <span>{playing ? "playing" : "paused"}</span>
            <span>{elapsedSeconds.toFixed(1)}s</span>
            <strong>{shownPanel ? shownReport.sentence : liveSummary}</strong>
          </>
        ) : (
          <strong>stored scores</strong>
        )}
      </div>
      <section className="ops-body">
        {side}
        {mode === "leaderboard" ? leaderboardMain : missionMain}
      </section>
    </div>
  );
}

export default function ComparisonApp() {
  const initial = useMemo(readQuery, []);
  const [mode, setMode] = useState<LabMode>(initial.mode);
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    fetchRuns().then(setRuns).catch(() => setRuns([]));
  }, []);

  const setModeAndUrl = (next: LabMode) => {
    setMode(next);
    const params = new URLSearchParams(window.location.search);
    params.set("mode", next);
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
  };

  return (
    <div className="lab comparison-lab">
      <header className="lab-header">
        <div>
          <h1 className="lab-title">AI Spacecraft Docking Lab</h1>
          <p className="lab-subtitle">Shared missions, stored scores, and a separate exploration sandbox.</p>
        </div>
        <nav className="mode-tabs" aria-label="Lab mode">
          <button className={mode === "leaderboard" ? "tab active" : "tab"} onClick={() => setModeAndUrl("leaderboard")} type="button">
            Leaderboard
          </button>
          <button className={mode === "mission" ? "tab active" : "tab"} onClick={() => setModeAndUrl("mission")} type="button">
            Mission
          </button>
          <button className={mode === "explore" ? "tab active" : "tab"} onClick={() => setModeAndUrl("explore")} type="button">
            Explore
          </button>
        </nav>
      </header>
      <main className={mode === "explore" ? "comparison-main explore-main" : "comparison-main"}>
        {mode === "explore" ? <Explorer active={mode === "explore"} runs={runs} /> : <BenchmarkWorkspace active mode={mode} />}
      </main>
    </div>
  );
}
