# Figure 11 (Section 6.6): Simulation Workflow UI Screenshot

Use this figure in **Chapter 6.6 – UI System and Operator Workflow** to make the backend/frontend integration concrete.

## What the figure must show

1. **Mission Map Workspace**  
   The central map where mission geometry/path is created and visualized.
2. **Mission Planning Tools**  
   The left control column used to connect SITL, arm/takeoff/land, select mode, and issue mission actions.
3. **Telemetry and Status Panels**  
   The right rail and status areas used for live telemetry, mission progress, and system health.

## Recommended capture setup

1. Start backend + frontend and open `/sim`.
2. Set browser window to 16:9 desktop view (for example 1920x1080) at 100% zoom.
3. Make sure the screenshot includes:
   - top status strip,
   - full left controls panel,
   - full center map,
   - full right telemetry panel.
4. Avoid modal dialogs/popups; keep the interface in a normal mission-monitoring state.
5. Save a raw screenshot (example):  
   `runs/figures/sim_ui_raw.png`

## Produce thesis-ready annotated Figure 11

```powershell
python scripts/annotate_simulation_ui_figure.py `
  --input runs/figures/sim_ui_raw.png `
  --output runs/figures/figure_6_6_simulation_workflow_ui.png `
  --output-pdf runs/figures/figure_6_6_simulation_workflow_ui.pdf
```

Important:
- Do **not** use `website_sim_report_*.png` or other generated chart/report images as the background.
- Figure 11 input must be a raw screenshot of the actual `/sim` UI page.
- The script now blocks known report-image filenames by default to prevent this mistake.

If your screenshot layout differs, adjust normalized callout boxes:

```powershell
python scripts/annotate_simulation_ui_figure.py `
  --input runs/figures/sim_ui_raw.png `
  --planning-box 0.01,0.17,0.23,0.78 `
  --map-box 0.25,0.17,0.49,0.78 `
  --telemetry-box 0.75,0.17,0.24,0.78
```

## Suggested thesis caption

**Figure 11. Simulation workflow UI used by the operator for mission planning, execution, and monitoring.**  
The interface combines (i) mission planning controls, (ii) a map-centered mission workspace, and (iii) telemetry/status panels connected to backend mission state and live vehicle streams.
