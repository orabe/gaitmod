# gaitmod desktop client (PySide6)

Desktop UI that talks to the FastAPI backend over HTTP.

## First-time setup (laptop)

```bash
cd /home/orabe/gaitmod/desktop_client
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (laptop)

```bash
cd /home/orabe/gaitmod/desktop_client
source .venv/bin/activate
export GAITMOD_API_BASE=http://localhost:8000
python app.py
```

## Backend (head node)

Start the FastAPI backend on the HPC head node and forward port 8000:

```bash
ssh -L 8000:localhost:8000 <user>@<hpc-head-node>
```

The app expects the API to respond at `http://localhost:8000`. You can also
change the API base URL inside the app header.

## Notes

- This app only calls the API; it does not run `sbatch` directly.
- Files open in your default browser via the API `/api/results/{run_id}/file` endpoint.
