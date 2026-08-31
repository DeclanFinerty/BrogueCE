#!/usr/bin/env python3
"""The naive approach, kept because the failure is instructive.

Plain subprocess pipes give the child no terminal, so ncurses cannot query a
window size, falls back to the terminfo default of 80x24, and Brogue parks on
its "needs a terminal window that is at least [100 x 34]" prompt forever.

The fix is not to drop subprocess -- it is to give the subprocess a real tty.
See brogue_harness.py, and docs/programmatic-control.md section 5.
"""

import subprocess
import time
from pathlib import Path

BIN_DIR = Path(__file__).resolve().parent.parent / "bin"

proc = subprocess.Popen(
    ["./brogue", "-t", "-n", "-s", "42"],
    cwd=BIN_DIR,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

time.sleep(2)
proc.stdin.write(b"l")  # move east
proc.stdin.flush()
time.sleep(2)

proc.terminate()
out, _ = proc.communicate(timeout=10)

print("exit code :", proc.returncode)
print("bytes out :", len(out))
print("output    :", out[:400])
print()
print("Expect the resize prompt above, not a dungeon. That is the point.")
