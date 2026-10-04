import { useEffect, useState } from "react";
import {
  fetchCheckpoints,
  fetchEval,
  fetchMetrics,
  fetchReplay,
  fetchRuns,
} from "./api";
import { SimCanvas } from "./SimCanvas";
import type {
  Checkpoint,
  EvalPoint,
  EvalSummary,
  Frame,
  Metrics,
  PoseBox,
  Replay,
  RunSummary,
  Selection,
  World,
} from "./types";
import "./App.css";

const RANDOM_CKPT: Checkpoint = { id: "random", label: "Random policy", steps: 0, path: null };
const SPEEDS = [1, 2, 4, 8] as const;

function formatArch(arch: number[] | undefined, label?: string): string {
  if (label) return label;
  if (!arch || arch.length === 0) return "MLP 64-64";
  return `MLP ${arch.join("-")}`;
}

function formatParams(n: number | undefined): string {
  if (!n) return "—";
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function formatSteps(steps: number): string {
  if (!steps) return "0";
  if (steps % 1000 === 0) return `${steps / 1000}k`;
  return steps.toLocaleString("en-US");
}

function formatEpoch(ck: Checkpoint): string {
  if (ck.id === "random") return "Random";
  if (ck.id.startsWith("final")) return "Final";
  return formatSteps(ck.steps);
}

function epochTitle(ck: Checkpoint): string {
  if (ck.id === "random") return "Random policy";
  if (ck.id.startsWith("final")) return "Final checkpoint";
  return `${ck.steps} steps`;
}

function formatFamily(family: string): string {
  if (family.includes("distance")) return "PPO 64 distance";
  if (family.startsWith("sac")) return "SAC 64";
  if (family.startsWith("td3")) return "TD3 64";
  if (family.startsWith("ddpg")) return "DDPG 64";
  if (family.includes("residual")) return "Residual MLP";
  if (family.includes("deeper")) return "PPO 128-128-64-32";
  if (family.includes("ppo_32")) return "PPO 32-32";
  if (family.includes("huber")) return "PPO Huber";
  if (family.includes("smooth")) return "PPO Smooth L1";
  if (family.includes("adamw")) return "PPO AdamW";
  if (family.includes("sgd")) return "PPO SGD";
  if (family.includes("cnn") || family.includes("pixel")) return "CNN pixels";
  if (family.includes("level2")) return "Level 2 spawn";
  if (family.includes("level3")) return "Level 3 velocity";
  if (family.includes("level4")) return "Level 4 rotate";
  if (family.includes("level5")) return "Level 5 wind";
  if (family.includes("level6")) return "Level 6 fuel";
  if (family.includes("level7")) return "Level 7 asteroids";
  if (family.includes("level8")) return "Level 8 moving";
  if (family.includes("wide")) return "PPO wide";
  if (family.includes("deep")) return "PPO deep";
  if (family.includes("ppo_64")) return "PPO 64";
  return family.replace(/_/g, " ");
}

function formatRunLabel(run: RunSummary): string {
  const family = run.family || (run.config_name || run.id).replace(/\.yaml$/i, "");
  const seed = Number.isFinite(run.seed) ? ` · seed ${run.seed}` : "";
  if (run.legacy || family === "ppo_mlp_static") return `Legacy 2-action${seed}`;
  if (family.includes("smoke")) return `Smoke MLP${seed}`;
  return `${formatFamily(family)}${seed}`;
}

const CURVE_COLORS = ["#e8a54b", "#3ee0c5", "#7eb6ff", "#d38bff"];
const BATTLE_COLORS = ["#7eb6ff", "#d38bff", "#3ee0c5", "#e85d4c"];

type CurvePoint = { x: number; y: number; low?: number; high?: number };

function CurveChart({
  series,
  markerX,
  label,
}: {
  series: { name: string; color: string; points: CurvePoint[] }[];
  markerX?: number | null;
  label: string;
}) {
  const all = series.flatMap((item) => item.points);
  if (all.length < 1) {
    return <p className="empty">Held-out eval appears after the first checkpoint.</p>;
  }
  const xs = all.map((point) => point.x);
  const ys = all.flatMap((point) => [point.y, point.low ?? point.y, point.high ?? point.y]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(0, ...ys);
  const maxY = Math.max(...ys, 1e-6);
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const xOf = (value: number) => ((value - minX) / spanX) * 280 + 10;
  const yOf = (value: number) => 58 - ((value - minY) / spanY) * 50;
  const marker = markerX != null && markerX >= minX && markerX <= maxX ? xOf(markerX) : null;
  return (
    <div>
      <svg className="chart" viewBox="0 0 300 64" role="img" aria-label={label}>
        {marker != null ? (
          <line x1={marker} x2={marker} y1="4" y2="60" stroke="#d6dde8" strokeDasharray="2 2" />
        ) : null}
        {series.map((item) => {
          if (item.points.length === 0) return null;
          if (item.points.length === 1) {
            const point = item.points[0];
            return <circle key={item.name} cx={xOf(point.x)} cy={yOf(point.y)} r="2.5" fill={item.color} />;
          }
          const line = item.points
            .map((point, i) => `${i === 0 ? "M" : "L"}${xOf(point.x).toFixed(1)},${yOf(point.y).toFixed(1)}`)
            .join(" ");
          const hasBand = item.points.some((point) => point.low != null && point.high != null && point.low !== point.high);
          const forward = item.points
            .map((point, i) => `${i === 0 ? "M" : "L"}${xOf(point.x).toFixed(1)},${yOf(point.high ?? point.y).toFixed(1)}`)
            .join(" ");
          const back = [...item.points]
            .reverse()
            .map((point) => `L${xOf(point.x).toFixed(1)},${yOf(point.low ?? point.y).toFixed(1)}`)
            .join(" ");
          return (
            <g key={item.name}>
              {hasBand ? <path d={`${forward} ${back}`} fill={item.color} opacity="0.18" /> : null}
              <path d={line} fill="none" stroke={item.color} strokeWidth="1.6" />
            </g>
          );
        })}
      </svg>
      <div className="chart-scale">
        <span>{maxY >= 10 ? maxY.toFixed(0) : maxY.toFixed(2)}</span>
        <span>{minY >= 10 ? minY.toFixed(0) : minY.toFixed(2)}</span>
      </div>
    </div>
  );
}

function familyBand(
  runs: RunSummary[],
  evals: Record<string, EvalSummary>,
  family: string,
  key: "success_rate" | "crash_rate" | "median_steps",
): CurvePoint[] {
  const grouped = new Map<number, number[]>();
  for (const run of runs.filter((item) => item.family === family)) {
    for (const point of evals[run.id]?.points ?? []) {
      const value = point[key];
      if (value == null) continue;
      const bucket = grouped.get(point.timesteps) ?? [];
      bucket.push(value);
      grouped.set(point.timesteps, bucket);
    }
  }
  return [...grouped.keys()].sort((a, b) => a - b).map((timesteps) => {
    const values = grouped.get(timesteps) ?? [];
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    return { x: timesteps, y: mean, low: Math.min(...values), high: Math.max(...values) };
  });
}

function toDeg(radians: number): number {
  return (radians * 180) / Math.PI;
}

function outcomeText(frame: Frame | undefined): { label: string; kind: string } {
  if (!frame) return { label: "idle", kind: "" };
  if (frame.success) return { label: "docked", kind: "success" };
  if (frame.out_of_fuel) return { label: "out of fuel", kind: "crash" };
  if (frame.hit_asteroid) return { label: "hit asteroid", kind: "crash" };
  if (frame.hit_hull) return { label: "hit station", kind: "crash" };
  if (frame.out_of_bounds) return { label: "left the map", kind: "crash" };
  if (frame.crash) return { label: "collision", kind: "crash" };
  if (frame.timeout) return { label: "timed out", kind: "" };
  return { label: "in flight", kind: "" };
}

function poseOr(
  pose: PoseBox | undefined,
  fallback: { cx: number; cy: number; w: number; h: number },
): PoseBox {
  return pose ?? { ...fallback, theta: 0 };
}

function insidePort(frame: Frame, world: World): boolean {
  const pose = poseOr(frame.port_pose, world.port);
  const dx = frame.x - pose.cx;
  const dy = frame.y - pose.cy;
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const lx = c * dx + s * dy;
  const ly = -s * dx + c * dy;
  return Math.abs(lx) <= pose.w / 2 && Math.abs(ly) <= pose.h / 2;
}

function meanOf(values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number => value != null);
  if (!present.length) return null;
  return present.reduce((sum, value) => sum + value, 0) / present.length;
}

function familyScores(runs: RunSummary[], evals: Record<string, EvalSummary>) {
  const families = [...new Set(runs.filter((run) => !run.legacy && run.family).map((run) => run.family as string))];
  return families.map((family) => {
    const latest: EvalPoint[] = [];
    for (const run of runs.filter((item) => item.family === family && !item.legacy)) {
      const points = evals[run.id]?.points ?? [];
      const point = points[points.length - 1];
      if (point) latest.push(point);
    }
    return {
      family,
      success: meanOf(latest.map((point) => point.success_rate)),
      fuel: meanOf(latest.map((point) => point.mean_fuel)),
      time: meanOf(latest.map((point) => point.mean_time)),
    };
  });
}

function formatPercent(value: number | null): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function formatMeasure(value: number | null, digits = 1): string {
  if (value == null) return "—";
  return value.toFixed(digits);
}

function RewardBars({ frame }: { frame: Frame | undefined }) {
  const items = [
    ["Distance", frame?.components.distance ?? 0],
    ["Velocity", frame?.components.velocity ?? 0],
    ["Rotation", frame?.components.rotation ?? 0],
    ["Fuel", frame?.components.fuel ?? 0],
    ["Time", frame?.components.time ?? 0],
    ["Docking", frame?.components.terminal ?? 0],
  ] as const;
  const maxAbs = Math.max(1, ...items.map(([, v]) => Math.abs(v)));
  return (
    <div className="bars">
      {items.map(([label, value]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div className="bar-track">
            <div
              className={`bar-fill ${value >= 0 ? "pos" : "neg"}`}
              style={{ width: `${(Math.abs(value) / maxAbs) * 100}%` }}
            />
          </div>
          <span>{value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function GateStrip({ frame, world }: { frame: Frame | undefined; world: World | null }) {
  const speedLimit = world?.dock_speed_max ?? 0.35;
  const angleLimit = world?.dock_angle_max_deg ?? 12;
  const speed = frame?.speed ?? 0;
  const heading = frame ? toDeg(frame.heading_error) : 0;
  return (
    <div className="stat-grid">
      <div className="stat">
        <span className="k">Distance</span>
        <span className="v">{(frame?.distance ?? 0).toFixed(2)}</span>
      </div>
      <div className={`stat ${frame && speed > speedLimit ? "bad" : ""}`}>
        <span className="k">Speed</span>
        <span className="v">
          {speed.toFixed(2)} / {speedLimit}
        </span>
      </div>
      <div className={`stat ${frame && heading > angleLimit ? "bad" : ""}`}>
        <span className="k">Heading</span>
        <span className="v">
          {heading.toFixed(1)}° / {Math.round(angleLimit)}°
        </span>
      </div>
      <div className="stat">
        <span className="k">Fuel</span>
        <span className="v">{(frame?.fuel ?? 0).toFixed(1)}</span>
      </div>
    </div>
  );
}

function EpisodeSummary({
  frame,
  successRate,
}: {
  frame: Frame | undefined;
  successRate: number;
}) {
  return (
    <div className="stat-grid">
      <div className="stat">
        <span className="k">Step</span>
        <span className="v">{frame?.t ?? 0}</span>
      </div>
      <div className="stat">
        <span className="k">Reward</span>
        <span className="v">{(frame?.reward_total ?? 0).toFixed(1)}</span>
      </div>
      <div className="stat">
        <span className="k">Fuel</span>
        <span className="v">{(frame?.fuel ?? 0).toFixed(1)}</span>
      </div>
      <div className="stat">
        <span className="k">Success window</span>
        <span className="v">{(successRate * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

function Inspector({
  selection,
  frame,
  frames,
  world,
}: {
  selection: Selection;
  frame: Frame | undefined;
  frames: Frame[];
  world: World | null;
}) {
  if (selection.kind === "trail") {
    const picked = frames[selection.index];
    const outcome = outcomeText(picked);
    return (
      <>
        <div className="stat-grid">
          <div className="stat">
            <span className="k">Step</span>
            <span className="v">{picked?.t ?? selection.index}</span>
          </div>
          <div className="stat">
            <span className="k">Position</span>
            <span className="v">
              {picked ? `${picked.x.toFixed(1)}, ${picked.y.toFixed(1)}` : "—"}
            </span>
          </div>
        </div>
        <p className="note">
          {picked?.done ? `This step ended the episode: ${outcome.label}.` : "A point on the path so far."}
        </p>
      </>
    );
  }

  if (selection.kind === "station") {
    return <p className="note">Touching this hull ends the episode.</p>;
  }

  if (!frame || !world) return null;

  if (selection.kind === "port") {
    const speedLimit = world.dock_speed_max ?? 0.35;
    const angleLimit = world.dock_angle_max_deg ?? 12;
    const rules = [
      ["Ship center inside the port", insidePort(frame, world)],
      [`Speed at most ${speedLimit}`, frame.speed <= speedLimit],
      [`Heading within ${Math.round(angleLimit)}° of the port`, toDeg(frame.heading_error) <= angleLimit],
    ] as const;
    return (
      <div className="rules">
        {rules.map(([label, ok]) => (
          <div className={`rule ${ok ? "ok" : "fail"}`} key={label}>
            <span>{label}</span>
            <span>{ok ? "met" : "not yet"}</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <>
      <div className="stat-grid">
        <div className="stat">
          <span className="k">Position</span>
          <span className="v">
            {frame.x.toFixed(1)}, {frame.y.toFixed(1)}
          </span>
        </div>
        <div className="stat">
          <span className="k">Speed</span>
          <span className="v">{frame.speed.toFixed(2)}</span>
        </div>
        <div className="stat">
          <span className="k">Heading error</span>
          <span className="v">{toDeg(frame.heading_error).toFixed(1)}°</span>
        </div>
        <div className="stat">
          <span className="k">Fuel</span>
          <span className="v">{frame.fuel.toFixed(1)}</span>
        </div>
        <div className="stat">
          <span className="k">Axial</span>
          <span className="v">{frame.axial.toFixed(2)}</span>
        </div>
        <div className="stat">
          <span className="k">Lateral</span>
          <span className="v">{frame.lateral.toFixed(2)}</span>
        </div>
        <div className="stat">
          <span className="k">Yaw</span>
          <span className="v">{frame.yaw.toFixed(2)}</span>
        </div>
      </div>
      <p className="note">
        Axial burns along the nose. Positive is forward, negative is the brake. Lateral strafes sideways. Yaw
        spins the ship.
      </p>
    </>
  );
}

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([RANDOM_CKPT]);
  const [checkpoint, setCheckpoint] = useState("random");
  const [seed, setSeed] = useState(42);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(2);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [evals, setEvals] = useState<Record<string, EvalSummary>>({});
  const [compare, setCompare] = useState<string[]>([]);
  const [battleIds, setBattleIds] = useState<string[]>([]);
  const [battleFrames, setBattleFrames] = useState<Record<string, Frame[]>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchRuns()
      .then((items) => {
        setRuns(items);
        if (items[0]) setRunId(items[0].id);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!runs.length) return;
    Promise.all(runs.map((run) => fetchEval(run.id).then((data) => [run.id, data] as const)))
      .then((pairs) => setEvals(Object.fromEntries(pairs)))
      .catch((err: Error) => setError(err.message));
  }, [runs]);

  useEffect(() => {
    if (!runId) {
      setCheckpoints([RANDOM_CKPT]);
      setCheckpoint("random");
      setMetrics(null);
      return;
    }
    Promise.all([fetchCheckpoints(runId), fetchMetrics(runId)])
      .then(([cks, mets]) => {
        setCheckpoints(cks.length ? cks : [RANDOM_CKPT]);
        setMetrics(mets);
        if (!cks.some((c) => c.id === checkpoint)) {
          setCheckpoint(cks[0]?.id ?? "random");
        }
      })
      .catch((err: Error) => setError(err.message));
  }, [runId]);

  useEffect(() => {
    let cancelled = false;
    setPlaying(false);
    setIndex(0);
    setSelection(null);
    fetchReplay({ run_id: runId || null, checkpoint, seed })
      .then((data) => {
        if (cancelled) return;
        setReplay(data);
        setIndex(0);
        setError(null);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, checkpoint, seed]);

  const frames = replay?.frames ?? [];
  const frame = frames[index];
  const selectedRun = runs.find((r) => r.id === runId);
  const dtMs = (replay?.dt ?? 0.05) * 1000;

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const timer = window.setInterval(() => {
      setIndex((prev) => {
        if (prev >= frames.length - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, dtMs / speed);
    return () => window.clearInterval(timer);
  }, [playing, frames.length, dtMs, speed]);

  useEffect(() => {
    const ids = battleIds.filter((id) => id !== runId);
    if (!ids.length) {
      setBattleFrames({});
      return;
    }
    let cancelled = false;
    const currentSteps = checkpoints.find((item) => item.id === checkpoint)?.steps;
    Promise.all(
      ids.map(async (id) => {
        const ghostCheckpoints = await fetchCheckpoints(id);
        const matched =
          currentSteps != null
            ? ghostCheckpoints.find((item) => item.id !== "random" && item.steps === currentSteps)
            : undefined;
        const ghostCheckpoint = matched?.id ?? (ghostCheckpoints.some((item) => item.id === "final") ? "final" : "random");
        const data = await fetchReplay({ run_id: id, checkpoint: ghostCheckpoint, seed });
        return [id, data.frames] as const;
      }),
    )
      .then((pairs) => {
        if (!cancelled) setBattleFrames(Object.fromEntries(pairs));
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [battleIds, runId, seed, checkpoint, checkpoints]);

  const outcome = outcomeText(frame);
  const meta = replay?.meta ?? {};
  const successRate = metrics?.success_rate_window ?? selectedRun?.success_rate_window ?? 0;
  const selectedEval = runId ? evals[runId] : undefined;
  const selectedCheckpoint = checkpoints.find((item) => item.id === checkpoint);
  const markerTimesteps =
    checkpoint === "final"
      ? selectedEval?.points[selectedEval.points.length - 1]?.timesteps
      : selectedCheckpoint && selectedCheckpoint.steps > 0 && selectedCheckpoint.steps < 1e12
        ? selectedCheckpoint.steps
        : null;
  const families = [...new Set(runs.filter((run) => !run.legacy && run.family).map((run) => run.family as string))];
  const scores = familyScores(runs, evals);
  const battleGhosts = battleIds.flatMap((id, ghostIndex) => {
    if (id === runId) return [];
    return [
      {
        frames: battleFrames[id] ?? [],
        color: BATTLE_COLORS[ghostIndex % BATTLE_COLORS.length],
      },
    ];
  });

  const handleSelect = (next: Selection | null) => {
    setSelection(next);
    if (next?.kind === "trail") {
      setPlaying(false);
      setIndex(next.index);
    }
  };

  const inspectorTitle =
    selection?.kind === "ship"
      ? "Ship"
      : selection?.kind === "port"
        ? "Port"
        : selection?.kind === "station"
          ? "Station"
          : selection?.kind === "trail"
            ? "Path"
            : "Episode";

  return (
    <div className="lab">
      <header className="lab-header">
        <div>
          <h1 className="lab-title">AI Spacecraft Docking Lab</h1>
          <p className="lab-subtitle">Same seed. Different checkpoints. Watch the approach.</p>
        </div>
        <div className="header-controls">
          <label>
            Run
            <select value={runId} onChange={(e) => setRunId(e.target.value)}>
              <option value="">no training run</option>
              {runs.map((run) => (
                <option key={run.id} value={run.id} title={run.id}>
                  {formatRunLabel(run)}
                </option>
              ))}
            </select>
          </label>
          <label className="seed-label">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </label>
        </div>
      </header>

      <div className="main">
        <section className="sim-pane">
          {frames.length === 0 ? (
            <p className="empty">Loading replay…</p>
          ) : (
            <SimCanvas
              world={replay?.world ?? null}
              frames={frames}
              index={index}
              ghosts={battleGhosts}
              selection={selection}
              onSelect={handleSelect}
            />
          )}
          {frame ? <div className={`outcome ${outcome.kind}`}>{outcome.label}</div> : null}
        </section>

        <aside className="side">
          <h2>Model</h2>
          <div className="meta-grid">
            <div className="meta">
              <span className="k">Algorithm</span>
              <span className="v">{meta.algorithm ?? selectedRun?.algorithm ?? "PPO"}</span>
            </div>
            <div className="meta">
              <span className="k">Architecture</span>
              <span className="v">{formatArch(meta.net_arch ?? selectedRun?.net_arch, meta.architecture)}</span>
            </div>
            <div className="meta">
              <span className="k">Parameters</span>
              <span className="v">{formatParams(meta.n_params ?? selectedRun?.n_params)}</span>
            </div>
            <div className="meta">
              <span className="k">Observation</span>
              <span className="v">{meta.obs_mode ?? selectedRun?.obs_mode ?? "state"}</span>
            </div>
            {meta.critic_loss ? (
              <div className="meta">
                <span className="k">Loss</span>
                <span className="v">{meta.critic_loss}</span>
              </div>
            ) : null}
            {meta.optimizer ? (
              <div className="meta">
                <span className="k">Optimizer</span>
                <span className="v">{meta.optimizer}</span>
              </div>
            ) : null}
            {meta.level ? (
              <div className="meta">
                <span className="k">Level</span>
                <span className="v">{meta.level}</span>
              </div>
            ) : null}
          </div>

          <h2>Approach</h2>
          <GateStrip frame={frame} world={replay?.world ?? null} />

          <h2>{inspectorTitle}</h2>
          {selection ? (
            <Inspector selection={selection} frame={frame} frames={frames} world={replay?.world ?? null} />
          ) : (
            <EpisodeSummary frame={frame} successRate={successRate} />
          )}

          <h2>Growth</h2>
          <p className="note">Held-out docks. The marker is the epoch on the scrubber.</p>
          <CurveChart
            label="Held-out success"
            markerX={markerTimesteps}
            series={[
              {
                name: "success",
                color: "#3ee0c5",
                points: (selectedEval?.points ?? []).map((point) => ({ x: point.timesteps, y: point.success_rate })),
              },
            ]}
          />
          <CurveChart
            label="Held-out crashes"
            markerX={markerTimesteps}
            series={[
              {
                name: "crash",
                color: "#e85d4c",
                points: (selectedEval?.points ?? []).map((point) => ({ x: point.timesteps, y: point.crash_rate })),
              },
            ]}
          />
          <CurveChart
            label="Median steps to dock"
            markerX={markerTimesteps}
            series={[
              {
                name: "steps",
                color: "#e8a54b",
                points: (selectedEval?.points ?? [])
                  .filter((point) => point.median_steps != null)
                  .map((point) => ({ x: point.timesteps, y: point.median_steps as number })),
              },
            ]}
          />

          <h2>Scoreboard</h2>
          <p className="note">Latest held-out missions for each family. Fuel and time are means.</p>
          {scores.length === 0 ? (
            <p className="empty">Scores appear after a training run.</p>
          ) : (
            <table className="matrix">
              <thead>
                <tr>
                  <th>Architecture</th>
                  <th>Success</th>
                  <th>Avg fuel</th>
                  <th>Avg time</th>
                </tr>
              </thead>
              <tbody>
                {scores.map((row) => (
                  <tr key={row.family}>
                    <td>{formatFamily(row.family)}</td>
                    <td>{formatPercent(row.success)}</td>
                    <td>{formatMeasure(row.fuel)}</td>
                    <td>{row.time == null ? "—" : `${row.time.toFixed(1)}s`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Compare</h2>
          <div className="compare">
            {families.map((family) => (
              <label key={family}>
                <input
                  type="checkbox"
                  checked={compare.includes(family)}
                  onChange={() => {
                    setCompare((current) => {
                      if (current.includes(family)) return current.filter((item) => item !== family);
                      if (current.length >= 4) return current;
                      return [...current, family];
                    });
                  }}
                />
                {formatFamily(family)}
              </label>
            ))}
          </div>
          <CurveChart
            label="Compared success"
            series={compare.map((family, index) => ({
              name: family,
              color: CURVE_COLORS[index % CURVE_COLORS.length],
              points: familyBand(runs, evals, family, "success_rate"),
            }))}
          />
          <CurveChart
            label="Compared crashes"
            series={compare.map((family, index) => ({
              name: family,
              color: CURVE_COLORS[index % CURVE_COLORS.length],
              points: familyBand(runs, evals, family, "crash_rate"),
            }))}
          />
          <h2>Battle</h2>
          <p className="note">Same seed on the canvas. Marks are the shared held-out missions.</p>
          <div className="compare">
            {runs
              .filter((run) => !run.legacy)
              .map((run) => (
                <label key={run.id}>
                  <input
                    type="checkbox"
                    checked={battleIds.includes(run.id)}
                    onChange={() => {
                      setBattleIds((current) => {
                        if (current.includes(run.id)) return current.filter((id) => id !== run.id);
                        if (current.length >= 4) return current;
                        return [...current, run.id];
                      });
                    }}
                  />
                  {formatRunLabel(run)}
                </label>
              ))}
          </div>
          {battleIds.map((id, battleIndex) => {
            const outcomes = evals[id]?.latest_outcomes ?? [];
            const color = BATTLE_COLORS[battleIndex % BATTLE_COLORS.length];
            const battleRun = runs.find((run) => run.id === id);
            return (
              <div className="battle-row" key={id}>
                <span className="battle-name" style={{ color }}>
                  {battleRun ? formatRunLabel(battleRun) : id}
                </span>
                {outcomes.length === 0 ? (
                  <span className="note">no held-out marks yet</span>
                ) : (
                  <div className="marks">
                    {outcomes.map((item) => (
                      <span
                        key={item.seed}
                        className={`mark ${item.success ? "ok" : "bad"}`}
                        title={`seed ${item.seed}`}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          <h2>Reward</h2>
          <RewardBars frame={frame} />
          {error ? <p className="error">{error}</p> : null}
        </aside>
      </div>

      <footer className="lab-footer">
        <label>
          Epoch
          <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            {checkpoints.map((ck) => (
              <option key={ck.id} value={ck.id} title={epochTitle(ck)}>
                {formatEpoch(ck)}
              </option>
            ))}
          </select>
        </label>
        <div className="playback">
          <button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
          <label>
            Speed
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value) as (typeof SPEEDS)[number])}
            >
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
            onChange={(e) => {
              setPlaying(false);
              setIndex(Number(e.target.value));
            }}
          />
          <span>
            {index}/{Math.max(0, frames.length - 1)}
          </span>
        </div>
      </footer>
    </div>
  );
}
