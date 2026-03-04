"""
Runner — Starts the main loop and the dashboard together.

Usage:
    python runner.py [--config configs/experiment_v1.yaml] [--port 8501]

Both processes share the same working directory and configuration file.
A KeyboardInterrupt (Ctrl-C) terminates both.
"""

import argparse
import multiprocessing
import subprocess
import sys


def _run_main(config: str) -> None:
    """Target for the main-loop process."""
    sys.argv = ["main.py", "--config", config]
    import main
    main.main()


def _run_retrain(config: str) -> None:
    """Target for the periodic retrainer process."""
    import logging
    import yaml
    from pathlib import Path
    from scripts.train import retrain_loop

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    cfg = yaml.safe_load(open(config))
    rt = cfg.get("retraining", {})
    retrain_loop(
        cfg=cfg,
        interval_s=int(rt.get("interval_seconds", 120)),
        output_path=Path(cfg["model"]["path"]),
        n_events=int(rt.get("n_events", 30_000)),
    )


def run(config: str = "configs/experiment_v1.yaml", port: int = 8501) -> None:
    """Launch the acquisition loop, the retrainer, and the dashboard in parallel."""
    main_proc = multiprocessing.Process(
        target=_run_main,
        args=(config,),
        name="main-loop",
        daemon=True,
    )

    retrain_proc = multiprocessing.Process(
        target=_run_retrain,
        args=(config,),
        name="retrain-loop",
        daemon=True,
    )

    dash_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", str(port),
            "--", "--config", config,
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    main_proc.start()
    retrain_proc.start()
    print(f"[runner] Main loop started  (pid {main_proc.pid})")
    print(f"[runner] Retrainer started  (pid {retrain_proc.pid})")
    print(f"[runner] Dashboard started  (pid {dash_proc.pid}, port {port})")

    try:
        main_proc.join()
    except KeyboardInterrupt:
        print("\n[runner] Interrupt received — shutting down…")
    finally:
        if main_proc.is_alive():
            main_proc.terminate()
            main_proc.join(timeout=5)
        if retrain_proc.is_alive():
            retrain_proc.terminate()
            retrain_proc.join(timeout=5)
        if dash_proc.poll() is None:
            dash_proc.terminate()
            try:
                dash_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dash_proc.kill()
        print("[runner] All processes stopped.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run main loop + dashboard together.")
    p.add_argument(
        "--config",
        default="configs/experiment_v1.yaml",
        help="Path to experiment YAML config",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501)",
    )
    return p.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    args = _parse_args()
    run(config=args.config, port=args.port)
