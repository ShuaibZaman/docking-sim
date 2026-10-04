export type Checkpoint = {
  id: string;
  label: string;
  steps: number;
  path: string | null;
};

export type RunSummary = {
  id: string;
  algorithm: string;
  net_arch: number[];
  n_params: number;
  obs_mode: string;
  seed: number;
  total_timesteps: number;
  config_name?: string;
  family?: string;
  legacy?: boolean;
  episode_count: number;
  success_rate_window: number;
};

export type RewardComponents = {
  distance?: number;
  velocity?: number;
  rotation?: number;
  fuel?: number;
  time?: number;
  terminal?: number;
};

export type PoseBox = {
  cx: number;
  cy: number;
  theta: number;
  w: number;
  h: number;
};

export type Asteroid = {
  x: number;
  y: number;
  r: number;
};

export type Frame = {
  t: number;
  x: number;
  y: number;
  theta: number;
  vx: number;
  vy: number;
  omega: number;
  fuel: number;
  distance: number;
  speed: number;
  heading_error: number;
  thrust: number;
  torque: number;
  axial: number;
  lateral: number;
  yaw: number;
  reward: number;
  reward_total: number;
  components: RewardComponents;
  done: boolean;
  success: boolean;
  crash: boolean;
  timeout: boolean;
  hit_hull: boolean;
  hit_asteroid?: boolean;
  out_of_bounds: boolean;
  out_of_fuel?: boolean;
  terminal_reason?: string;
  absolute_speed?: number;
  relative_vx?: number;
  relative_vy?: number;
  station?: PoseBox;
  port_pose?: PoseBox;
  asteroids?: Asteroid[];
};

export type Selection =
  | { kind: "ship" }
  | { kind: "port" }
  | { kind: "station" }
  | { kind: "trail"; index: number };

export type World = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  ship_radius: number;
  hull: { cx: number; cy: number; w: number; h: number };
  port: { cx: number; cy: number; w: number; h: number };
  dock_speed_max: number;
  dock_angle_max_deg: number;
  approach_angle: number;
};

export type Replay = {
  run_id: string | null;
  checkpoint: string;
  baseline?: string | null;
  seed: number;
  scenario_id?: string | null;
  scenario_hash?: string | null;
  success: boolean;
  crash: boolean;
  timeout: boolean;
  steps: number;
  reward_total: number;
  dt: number;
  world: World;
  meta: {
    algorithm?: string;
    policy?: string;
    net_arch?: number[];
    n_params?: number;
    obs_mode?: string;
    architecture?: string;
    critic_loss?: string;
    optimizer?: string;
    level?: number | null;
  };
  frames: Frame[];
};

export type EpisodeMetric = {
  episode: number;
  timesteps: number;
  reward: number;
  length: number;
  success: number;
  crash: number;
  fuel: number;
  distance: number;
};

export type Metrics = {
  episode_count: number;
  success_rate_window: number;
  episodes: EpisodeMetric[];
};

export type EvalPoint = {
  timesteps: number;
  checkpoint: string;
  success_rate: number;
  crash_rate: number;
  median_steps: number | null;
  mean_fuel?: number | null;
  mean_time?: number | null;
  mean_fuel_success?: number | null;
  mean_time_success?: number | null;
  success_ci_low?: number | null;
  success_ci_high?: number | null;
  failure_reasons?: Record<string, number>;
  strata?: Array<{ stratum: string; n: number; success_rate: number; crash_rate: number }>;
  n: number;
};

export type EvalOutcome = {
  seed: number;
  success: boolean;
  crash: boolean;
  fuel_used: number | null;
  time: number | null;
  scenario_id?: string | null;
  scenario_hash?: string | null;
  stratum?: string | null;
  terminal_reason?: string;
};

export type EvalSummary = {
  id: string;
  family: string;
  legacy: boolean;
  seed: number;
  points: EvalPoint[];
  latest_outcomes?: EvalOutcome[];
};

export type BenchmarkScenario = {
  id: string;
  benchmark_id: string;
  protocol_version: string;
  seed: number;
  stratum: string;
  scenario_hash: string;
  initial_state: Record<string, number>;
  station_pose: Record<string, number>;
  asteroids: Array<{ x: number; y: number; r: number }>;
  disturbance_trace: Array<{ x: number; y: number }>;
};

export type Benchmark = {
  id: string;
  label: string;
  description: string;
  protocol_version: string;
  n_scenarios: number;
  scene_fingerprint: string;
  interface_fingerprint: string;
  strata: string[];
  scenarios?: BenchmarkScenario[];
};

export type BenchmarkCandidate = {
  id: string;
  label: string;
  algorithm?: string;
  architecture?: string | null;
  training_seed?: number | null;
  baseline?: string;
  description?: string;
  compatible: boolean;
  reason: string | null;
  checkpoints: Checkpoint[];
};

export type CandidateSelection = {
  run_id?: string | null;
  checkpoint: string;
  baseline?: string | null;
};

export type BenchmarkPanel = {
  candidate: CandidateSelection;
  label: string;
  replay: Replay;
};

export type BenchmarkReplay = {
  benchmark: Benchmark;
  scenario: BenchmarkScenario;
  panels: BenchmarkPanel[];
};

export type ShowcasePreset = {
  id: string;
  label: string;
  description: string;
  benchmark_id: string;
  scenario_id: string;
  stratum: string;
  scenario_hash: string;
};

export type PairwiseScenario = {
  scenario_id: string;
  stratum?: string | null;
  left_success: boolean;
  right_success: boolean;
  left_terminal_reason?: string | null;
  right_terminal_reason?: string | null;
};

export type PairwiseSummary = {
  benchmark_id: string;
  n_common: number;
  outcomes: {
    both_success: number;
    left_only: number;
    right_only: number;
    neither: number;
  };
  mean_fuel_delta_left_minus_right: number | null;
  mean_time_delta_left_minus_right: number | null;
  scenarios?: PairwiseScenario[];
};

export type BenchmarkCompare = {
  benchmark_id: string;
  pairwise: Array<{
    left: CandidateSelection;
    right: CandidateSelection;
    summary: PairwiseSummary;
  }>;
};
