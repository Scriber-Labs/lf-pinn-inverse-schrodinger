# Data Directory
This directory stores curated run artifacts used for demos, validation notebooks, and a future Dash app.

Generated training runs are written to:
```text
artifacts/runs/<run_id>
```

Selected runs may be promoted into:
```text
data/reference_runs/<run_id>/
```

A single reference run directory is expected to contain:
- `config.json`
- `history.json`
- `diagnostics.npz`
- `ground_truth.pt`
- `model.pt`
- `run_id.txt`
- optionally, 
    - `pod_metrics.csv`
    - `training_analysis.csv`

The SQLite database `training_runs.db` and latest-run pointer `run_id.txt` are local generated files and are not required for loading reference runs.