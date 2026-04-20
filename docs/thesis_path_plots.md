# Thesis Drone Path Plots

Use `scripts/plot_drone_paths.py` to generate publication-ready trajectory figures from bridge run artifacts.

## Install

```bash
pip install matplotlib
```

## 1) Planned Path vs Flown Path (single run)

```bash
python -m scripts.plot_drone_paths single \
  --run-dir runs/ardupilot_scan/<run_timestamp_profile> \
  --label "Scan Task - Single Flight" \
  --out runs/figures/scan_single_path.png
```

This figure is strong for:
- scan task trajectory quality
- waypoint-following fidelity
- one representative successful or failed run

## 2) Successful vs Failed Trajectory (side-by-side)

```bash
python -m scripts.plot_drone_paths success-failure \
  --success-run-dir runs/ardupilot_scan/<successful_run_dir> \
  --failure-run-dir runs/ardupilot_scan/<failed_run_dir> \
  --title "Scan Task: Success vs Failure" \
  --out runs/figures/scan_success_vs_failure.png
```

This figure is strong for:
- showing failure mode structure
- showing why your controller improves robustness

## 3) Sim vs Real Overlay

```bash
python -m scripts.plot_drone_paths compare \
  --sim-run-dir runs/ardupilot_scan/<sim_run_dir> \
  --real-run-dir runs/ardupilot_scan/<real_run_dir> \
  --sim-label "Simulation" \
  --real-label "Real Flight" \
  --title "Sim-to-Real Path Consistency" \
  --out runs/figures/sim_vs_real_overlay.png
```

This figure is strong for:
- sim-to-real transfer claim
- geometric path consistency

## 4) Website Simulation Run -> Informative Report Graphs

Start a tracked run in backend:

```bash
curl -X POST http://127.0.0.1:8000/api/runs/start -H "Content-Type: application/json" -d "{}"
```

Run the simulation mission from the website UI, then export:

```bash
curl http://127.0.0.1:8000/api/runs/current/export/json -o runs/figures/website_run_export.json
```

Generate multi-panel report (path + speed/altitude + coverage intensity):

```bash
python -m scripts.plot_drone_paths website-report \
  --run-export-json runs/figures/website_run_export.json \
  --title "Website Sim Run Report" \
  --out runs/figures/website_sim_report.png
```

Optional: stop run tracking:

```bash
curl -X POST http://127.0.0.1:8000/api/runs/stop
```

## Notes

- Input expects bridge-style `telemetry.jsonl` and optional `summary.json`.
- `single` infers planned target sequence from `target_update_flag` rows in telemetry.
- Figure status labels use `summary.json` (`exit_reason`, `final_coverage`) when available.
- If `telemetry.jsonl` is not in the run dir, pass `--telemetry` and optionally `--summary` explicitly.
- You can also use run-export JSON directly:
  - `single --run-export-json ...`
  - `compare --sim-run-export-json ... --real-run-export-json ...`
  - `success-failure --success-run-export-json ... --failure-run-export-json ...`
