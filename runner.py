"""
Runner — starts main.main() and the Streamlit dashboard together.

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


def run(config: str = "configs/experiment_v1.yaml", port: int = 8501) -> None:
    """Launch the loop and the Streamlit dashboard in parallel."""
    main_proc = multiprocessing.Process(
        target=_run_main,
        args=(config,),
        name="main-loop",
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
    print(f"[runner] Main loop started  (pid {main_proc.pid})")
    print(f"[runner] Dashboard started          (pid {dash_proc.pid}, port {port})")

    try:
        main_proc.join()
    except KeyboardInterrupt:
        print("\n[runner] Interrupt received — shutting down…")
    finally:
        if main_proc.is_alive():
            main_proc.terminate()
            main_proc.join(timeout=5)
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
