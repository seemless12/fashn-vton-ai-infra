"""
auto_idle_watchdog.py
---------------------
Automatic Idle Shutdown Watchdog for Shopping Buddy AI GPU Server.
Monitors inference activity on the EC2 GPU instance.
If no try-on jobs have been processed for IDLE_TIMEOUT_MINUTES (default: 10 minutes),
it automatically executes a graceful system shutdown (poweroff).

AWS automatically transitions the EC2 instance from 'running' to 'stopped' state,
reducing GPU compute billing to exactly $0.00/hr.

Usage:
  - Run as a systemd service or background daemon:
      python auto_idle_watchdog.py --daemon
  - One-off check (e.g. from cron every 2 minutes):
      python auto_idle_watchdog.py --check
"""

import argparse
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("idle_watchdog")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

LOCKFILE = "/tmp/last_fashn_job"
DEFAULT_IDLE_MINUTES = 10


def get_idle_seconds() -> float:
    """Returns seconds since last job, or seconds since system boot if no jobs yet."""
    now = time.time()
    if os.path.exists(LOCKFILE):
        try:
            mtime = os.path.getmtime(LOCKFILE)
            return now - mtime
        except Exception:
            pass

    # Fallback to system uptime
    try:
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.readline().split()[0])
            return uptime_seconds
    except Exception:
        return 0.0


def touch_job():
    """Touch the lockfile to mark recent job activity."""
    try:
        with open(LOCKFILE, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        logger.error(f"Failed to touch lockfile {LOCKFILE}: {e}")


def is_gpu_busy() -> bool:
    """Checks nvidia-smi GPU utilization to avoid shutting down during an active render."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if res.returncode == 0:
            util = int(res.stdout.strip())
            return util > 15  # GPU is actively computing
    except Exception:
        pass
    return False


def shutdown_instance(dry_run: bool = False):
    """Executes system shutdown to stop EC2 billing."""
    logger.warning("=" * 60)
    logger.warning("IDLE TIMEOUT EXCEEDED! Auto-stopping GPU instance to save costs.")
    logger.warning("Compute billing will drop to $0.00/hr.")
    logger.warning("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would execute: sudo shutdown -h now")
        return

    # Trigger graceful system shutdown
    subprocess.run(["sudo", "shutdown", "-h", "now"])


def run_check(idle_minutes: int, dry_run: bool = False):
    """Performs a single idle check."""
    idle_secs = get_idle_seconds()
    max_idle_secs = idle_minutes * 60
    logger.info(f"Current idle time: {idle_secs:.1f}s / max allowed: {max_idle_secs}s ({idle_minutes} min)")

    if idle_secs > max_idle_secs:
        if is_gpu_busy():
            logger.info("GPU is currently busy. Postponing shutdown.")
            touch_job()
            return

        shutdown_instance(dry_run=dry_run)
    else:
        logger.info(f"System active. {max_idle_secs - idle_secs:.1f}s remaining before auto-sleep.")


def run_daemon(idle_minutes: int, poll_interval: int = 60, dry_run: bool = False):
    """Runs continuous watchdog loop."""
    logger.info(f"Starting Shopping Buddy Auto-Idle Watchdog (Timeout: {idle_minutes} min, Poll: {poll_interval}s)")
    touch_job()  # Initialize on startup

    while True:
        try:
            run_check(idle_minutes, dry_run=dry_run)
        except Exception as e:
            logger.error(f"Watchdog check failed: {e}")
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shopping Buddy Auto-Idle Shutdown Watchdog")
    parser.add_argument("--timeout", type=int, default=DEFAULT_IDLE_MINUTES, help="Idle timeout in minutes")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous daemon")
    parser.add_argument("--check", action="store_true", help="Perform single check and exit")
    parser.add_argument("--touch", action="store_true", help="Touch lockfile to refresh idle timer")
    parser.add_argument("--dry-run", action="store_true", help="Simulate shutdown without executing")

    args = parser.parse_args()

    if args.touch:
        touch_job()
        print("Touched lockfile. Idle timer reset.")
    elif args.daemon:
        run_daemon(args.timeout, dry_run=args.dry_run)
    else:
        run_check(args.timeout, dry_run=args.dry_run)
