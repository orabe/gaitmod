# gaitmod config editor (PySide6)

Local desktop UI that loads/saves hyperparameter JSON configs with form controls,
plus a basic local training launcher.

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
python app.py
```

If the config directory is different on your machine, click Browse and select
the folder that contains your `*.json` config files.

## Notes

- Training runs `python gaitmod/train.py --hyperparams-config <path>` in the repo root.
- Form controls are inferred from existing configs and `gaitmod/train.py` constants.
