from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick deterministic sanity check for production scan model.")
    parser.add_argument("--model", type=str, default="runs/production_scan/best_model.zip")
    parser.add_argument("--preset", type=str, default="A2")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="scan")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional explicit output path. Defaults to runs/production_scan/metrics/verify_*.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.json_out.strip():
        out_path = Path(args.json_out)
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("runs/production_scan/metrics") / f"verify_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scripts.eval",
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--preset",
        str(args.preset),
        "--episodes",
        str(int(args.episodes)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--json-out",
        str(out_path),
        "--cfg-override",
        "scan_debug_oob=false",
    ]
    subprocess.run(cmd, check=True)

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    payload = {
        "ok": True,
        "model_path": str(args.model),
        "preset": str(args.preset),
        "task": str(args.task),
        "episodes": int(summary.get("episodes", args.episodes)),
        "success_count": int(summary.get("success_count", 0)),
        "crash_count": int(summary.get("crash_count", 0)),
        "coverage_mean": float(summary.get("coverage_mean", 0.0)),
        "json": str(out_path),
    }
    wrapped = {
        "verify": payload,
        "eval_summary": summary,
    }
    out_path.write_text(json.dumps(wrapped, indent=2), encoding="utf-8")
    print("[verify_production_scan]", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
