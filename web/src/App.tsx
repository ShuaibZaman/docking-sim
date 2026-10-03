import { useEffect, useState } from "react";
import { fetchCheckpoints, fetchMetrics, fetchReplay, fetchRuns } from "./api";
import { SimCanvas } from "./SimCanvas";
import type { Checkpoint, Frame, Metrics, Replay, RunSummary, Selection, World } from "./types";
import "./App.css";

const RANDOM_CKPT: Checkpoint = { id: "random", label: "Random policy", steps: 0, path: null };
const SPEEDS = [1, 2, 4, 8] as const;

function formatArch(arch: number[] | undefined): string {
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

function formatRunLabel(run: RunSummary): string {
  const steps = formatSteps(run.total_timesteps);
  const source = (run.config_name || run.id).replace(/\.yaml$/i, "");
  const slug = source.replace(/^\d{8}_\d{6}_/, "");
  let name = slug.replace(/_/g, " ");
  if (slug.includes("static")) name = "Static MLP";
  else if (slug.includes("smoke")) name = "Smoke MLP";
  return `${name} · ${steps}`;
}

function toDeg(radians: number): number {
  return (radians * 180) / Math.PI;
}

function outcomeText(frame: Frame | undefined): { label: string; kind: string } {
  if (!frame) return { label: "idle", kind: "" };
  if (frame.success) return { label: "docked", kind: "success" };
  if (frame.hit_hull) return { label: "hit station", kind: "crash" };
  if (frame.out_of_bounds) return { label: "left the map", kind: "crash" };
  if (frame.crash) return { label: "collision", kind: "crash" };
  if (frame.timeout) return { label: "timed out", kind: "" };
  return { label: "in flight", kind: "" };
}

function insidePort(frame: Frame, world: World): boolean {
  const { cx, cy, w, h } = world.port;
  return frame.x >= cx - w / 2 && frame.x <= cx + w / 2 && frame.y >= cy - h / 2 && frame.y <= cy + h / 2;
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

function RewardChart({ metrics, mark }: { metrics: Metrics | null; mark?: number }) {
  const points = metrics?.episodes ?? [];
  if (points.length < 2) {
    return <p className="empty">Train a run to plot episode reward.</p>;
  }
  const rewards = points.map((p) => p.reward);
  const min = Math.min(...rewards);
  const max = Math.max(...rewards);
  const span = Math.max(1e-6, max - min);
  const d = rewards
    .map((r, i) => {
      const x = (i / (rewards.length - 1)) * 300;
      const y = 58 - ((r - min) / span) * 50;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const markY =
    mark !== undefined && mark >= min && mark <= max ? 58 - ((mark - min) / span) * 50 : null;
  return (
    <div>
      <svg className="chart" viewBox="0 0 300 64" role="img" aria-label="Episode reward">
        {markY !== null ? (
          <line x1="0" x2="300" y1={markY} y2={markY} stroke="#3ee0c5" strokeDasharray="3 3" strokeWidth="1" />
        ) : null}
        <path d={d} fill="none" stroke="#e8a54b" strokeWidth="1.6" />
      </svg>
      <div className="chart-scale">
        <span>{max.toFixed(0)}</span>
        <span>{min.toFixed(0)}</span>
      </div>
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
      [`Heading within ${Math.round(angleLimit)}° of straight up`, toDeg(frame.heading_error) <= angleLimit],
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
          <span className="k">Thrust</span>
          <span className="v">{frame.thrust.toFixed(2)}</span>
        </div>
        <div className="stat">
          <span className="k">Torque</span>
          <span className="v">{frame.torque.toFixed(2)}</span>
        </div>
      </div>
      <p className="note">Thrust points along the nose. Torque spins the ship.</p>
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

  const outcome = outcomeText(frame);
  const meta = replay?.meta ?? {};
  const successRate = metrics?.success_rate_window ?? selectedRun?.success_rate_window ?? 0;

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
              <span className="v">{formatArch(meta.net_arch ?? selectedRun?.net_arch)}</span>
            </div>
            <div className="meta">
              <span className="k">Parameters</span>
              <span className="v">{formatParams(meta.n_params ?? selectedRun?.n_params)}</span>
            </div>
            <div className="meta">
              <span className="k">Observation</span>
              <span className="v">{meta.obs_mode ?? selectedRun?.obs_mode ?? "state"}</span>
            </div>
          </div>

          <h2>Approach</h2>
          <GateStrip frame={frame} world={replay?.world ?? null} />

          <h2>{inspectorTitle}</h2>
          {selection ? (
            <Inspector selection={selection} frame={frame} frames={frames} world={replay?.world ?? null} />
          ) : (
            <EpisodeSummary frame={frame} successRate={successRate} />
          )}

          <h2>Reward</h2>
          <RewardChart metrics={metrics} mark={replay?.reward_total} />
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
