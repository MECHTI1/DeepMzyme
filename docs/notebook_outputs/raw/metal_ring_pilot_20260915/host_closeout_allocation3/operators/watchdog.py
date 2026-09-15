"""Named teardown watchdog using the remaining original ten-hour cap."""
from pathlib import Path
import subprocess
import sys
import time
from ring_common import allocation_deadline, read, save


def main():
    ops = Path(__file__).resolve().parent
    config = read(ops/"session_config.json")
    deadline = allocation_deadline(config, stop_margin=300)
    while not (ops/"session_stopped.json").exists():
        remaining = deadline-time.time()
        if remaining <= 0:
            try:
                result = subprocess.run([sys.executable, str(ops/"ring_host.py"), "stop"],
                                        capture_output=True, text=True, timeout=140)
                event = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            except Exception as exc:
                event = dict(error_type=type(exc).__name__)
            save(ops/"watchdog_last_attempt.json", dict(event, epoch=time.time()))
            if not (ops/"session_stopped.json").exists():
                time.sleep(10)
        else:
            time.sleep(min(30, remaining))


if __name__ == "__main__":
    main()
