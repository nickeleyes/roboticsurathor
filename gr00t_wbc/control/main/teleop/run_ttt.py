"""Launch isolated control and task processes for G1 tic-tac-toe."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable


_SHUTDOWN_TIMEOUT_SECONDS = 8.0


def split_cpu_affinity(allowed_cpus: Iterable[int]) -> tuple[set[int], set[int]]:
    """Split the CPUs available to this launcher into disjoint child CPU sets."""
    cpus = sorted(set(allowed_cpus))
    if not cpus:
        raise RuntimeError("The launcher has no CPUs in its affinity mask")
    if len(cpus) == 1:
        return {cpus[0]}, {cpus[0]}

    control_count = (len(cpus) + 1) // 2
    return set(cpus[:control_count]), set(cpus[control_count:])


def _format_cpu_set(cpus: set[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def _child_setup(cpus: set[int]) -> None:
    """Isolate a child before it imports any controller or vision modules."""
    os.sched_setaffinity(0, cpus)


def _stop_child(child: subprocess.Popen, sig: int) -> None:
    if child.poll() is not None:
        return
    try:
        child.send_signal(sig)
    except ProcessLookupError:
        pass


def _shutdown(children: list[subprocess.Popen], sig: int = signal.SIGTERM) -> None:
    for child in children:
        _stop_child(child, sig)

    deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
    for child in children:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            break

    survivors = [child for child in children if child.poll() is None]
    for child in survivors:
        _stop_child(child, signal.SIGKILL)
    for child in survivors:
        child.wait()


def main(argv: list[str] | None = None) -> int:
    forwarded_args = sys.argv[1:] if argv is None else argv
    script_dir = Path(__file__).resolve().parent
    child_specs = (
        ("control", script_dir / "run_ttt_control.py"),
        ("task", script_dir / "run_ttt_task.py"),
    )

    allowed_cpus = os.sched_getaffinity(0)
    control_cpus, task_cpus = split_cpu_affinity(allowed_cpus)
    cpu_sets = {"control": control_cpus, "task": task_cpus}
    if control_cpus == task_cpus:
        print(
            "WARNING: only one CPU is available; processes are separate but CPU "
            "contention cannot be prevented.",
            flush=True,
        )

    print(
        "TTT process isolation: "
        f"control CPUs [{_format_cpu_set(control_cpus)}], "
        f"task/vision CPUs [{_format_cpu_set(task_cpus)}]",
        flush=True,
    )

    children: list[subprocess.Popen] = []
    shutdown_signal: int | None = None

    def handle_signal(signum, _frame) -> None:
        nonlocal shutdown_signal
        shutdown_signal = signum

    old_handlers = {
        sig: signal.signal(sig, handle_signal)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }

    try:
        for name, script in child_specs:
            command = [sys.executable, str(script), *forwarded_args]
            cpus = cpu_sets[name]
            child = subprocess.Popen(
                command,
                preexec_fn=lambda cpus=cpus: _child_setup(cpus),
            )
            children.append(child)
            print(f"Started TTT {name} process (pid {child.pid})", flush=True)

        # Keep this lightweight supervisor off the CPUs reserved for control too.
        os.sched_setaffinity(0, task_cpus)

        while shutdown_signal is None:
            for child in children:
                return_code = child.poll()
                if return_code is not None:
                    print(
                        f"TTT child pid {child.pid} exited with code {return_code}; "
                        "stopping the other process.",
                        flush=True,
                    )
                    _shutdown(children)
                    return return_code
            time.sleep(0.1)

        print(f"Stopping TTT processes after signal {shutdown_signal}", flush=True)
        _shutdown(children, shutdown_signal)
        return 0
    finally:
        if any(child.poll() is None for child in children):
            _shutdown(children)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        os.sched_setaffinity(0, allowed_cpus)


if __name__ == "__main__":
    raise SystemExit(main())
