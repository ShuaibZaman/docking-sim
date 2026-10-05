# AI Spacecraft Docking Lab

2D docking simulator, multi-algorithm RL training, and a comparison dashboard. Benchmarks compare **shared missions**. Custom seeds stay in a separate Explore tab.

You need **two processes**: a FastAPI replay server (port **8000**) and the Vite UI (port **5173**).

## One-time setup

From the repo root, in PowerShell. Use **Python 3.12** (not 3.14; Torch 2.11 currently fails to import on 3.13.8 on this machine).

```powershell
py -3.12 -m venv .venv
# If `py -3.12` is missing:
# & "$env:APPDATA\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe" -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

cd web
npm install
cd ..
```

A short smoke run already exists under `artifacts/runs/` after the first training pass. If you have no runs yet, start the app anyway (Random policy and the PD baseline work) or train first — see below.

## Start

Open **two** PowerShell windows. Both commands are from the **repo root**.

**Terminal 1 — API**

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Leave this running. You should see `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2 — dashboard**

```powershell
cd web
npm run dev
```

Leave this running. You should see `Local: http://localhost:5173/`.

Open [http://localhost:5173](http://localhost:5173). The lab has three surfaces:

**Leaderboard** ranks compatible trained runs on the selected suite. Success is better when higher. Crash rate, fuel on success, and time on success are better when lower. Each success rate includes its stored confidence interval. Learning curves and stratum or terminal-reason breakdowns use the scores already written to `eval.jsonl`. Pinning PD or Random does not invent a ranking: deltas against that baseline appear only after it has stored scores.

**Mission** compares two to four candidates on one shared mission. The viewports stay in front; suite, stratum, and checkpoint controls sit in the side column. A status strip keeps the suite, mission, stratum, playback, and outcome visible. Each panel has its own scene, capture gates, and trajectory, attitude, and propulsion readouts. The approach corridor, capture box, hull keep-out, and target-frame inset are drawn on the canvas. Space plays or pauses, the arrow keys scrub, and 1–4 highlight a panel. Checkpoints stay explicit. Incompatible contracts stay collapsed. **Export JSON** saves the exact replay bundle. The address bar keeps the suite, mission, and candidate checkpoints, so a reload restores the same comparison.

**Explore** is a custom-seed sandbox for a single run. It uses the same scene overlays and telemetry, and it is kept off the leaderboard because a typed seed is not a persisted shared mission.

A dock counts only when the ship is inside the active port, nearly stopped (relative speed and spin at or below 0.08), and facing that port's approach. Spin costs reward on every step, including far from the station. The first capture of a two-port level is a one-second hold, then the target switches.

Explore can preview levels 1–11 without training: close static, the existing curriculum, a long approach, corridor obstacles, a moving port, and the two-port transfer. The canvas fills the page. Model details and the plain-language capture reason sit in the side rail.

Use **Quick 20** while iterating and **Canonical 100** for a portfolio comparison. Showcase presets (`straight-in`, `lateral-offset`, `drifting`, `precision`) jump to named missions. Mission comparison keeps previous and next for the shared list, one inspector for the highlighted panel, and stored curves behind Evidence.

## Stop

In **each** of the two terminals:

1. Click the terminal so it is focused.
2. Press **Ctrl+C**.
3. If it asks to terminate the batch job, confirm with **Y**.

The API window should leave port 8000; the `npm run dev` window should leave port 5173.

### If a port is still in use

Someone else may still be bound to 8000 or 5173 (process killed without Ctrl+C, or a leftover from Cursor). From any PowerShell:

```powershell
# See what owns the ports
Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue |
  Select-Object LocalPort, OwningProcess, State

# Stop those processes
Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
```

Then start again with the two commands in **Start**.

To stop a **training** run, focus that terminal and press **Ctrl+C**. Partial checkpoints that already flushed to `artifacts/runs/<run_id>/checkpoints/` stay on disk. Resume with the original config:

```powershell
.\.venv\Scripts\python.exe -m docking_sim.training.train --config configs\ppo_64.yaml --resume <run_id> --additional-timesteps 20000
```

Resume refuses a different config or scene contract.

## Train (optional)

From the repo root. This is a separate process from the dashboard; stop it with Ctrl+C.

Smoke run (seconds, good for wiring the UI):

```powershell
.\.venv\Scripts\python.exe -m docking_sim.training.train --config configs\ppo_mlp_smoke.yaml --name smoke
```

Default static-station PPO (~200k steps, not overnight):

```powershell
.\.venv\Scripts\python.exe -m docking_sim.training.train --config configs\ppo_mlp_static.yaml
```

Opt-in robustness and orbital-relative experiments (do not start these unless you intend to train):

- `configs/ppo_64_constrained.yaml` — safety-constrained reward
- `configs/ppo_64_actuator.yaml` — actuator lag, scale uncertainty, persistent wind
- `configs/ppo_64_sensors.yaml` — noisy, delayed observations
- `configs/ppo_orbital.yaml` — Hill/Clohessy–Wiltshire relative dynamics

```powershell
.\.venv\Scripts\python.exe -m docking_sim.training.sweep --suite robustness --dry-run
```

Runs land in `artifacts/runs/<timestamp>_<name>/`. Refresh the dashboard (or restart the API if a run appeared while it was already up — a refresh is enough; the API reads the folder on each request).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
