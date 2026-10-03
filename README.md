# AI Spacecraft Docking Lab

2D docking simulator, PPO training, and a replay dashboard. Same seed, different checkpoints — watch the approach change.

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

A short smoke run already exists under `artifacts/runs/` after the first training pass. If you have no runs yet, start the app anyway (Random policy works) or train first — see below.

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

Open [http://localhost:5173](http://localhost:5173). Pick a run, keep the seed fixed, hit **Play**, and switch **Random → checkpoints** to compare the same starting condition.

Do not close those terminals while you use the lab. Closing them (or Ctrl+C) stops the servers.

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

To stop a **training** run, focus that terminal and press **Ctrl+C**. Partial checkpoints that already flushed to `artifacts/runs/<run_id>/checkpoints/` stay on disk.

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

Runs land in `artifacts/runs/<timestamp>_<name>/`. Refresh the dashboard (or restart the API if a run appeared while it was already up — a refresh is enough; the API reads the folder on each request).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
