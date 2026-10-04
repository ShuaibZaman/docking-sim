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
  out_of_bounds: boolean;
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
  seed: number;
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
  n: number;
};

export type EvalSummary = {
  id: string;
  family: string;
  legacy: boolean;
  seed: number;
  points: EvalPoint[];
};
