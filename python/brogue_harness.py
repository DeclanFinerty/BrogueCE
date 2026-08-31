#!/usr/bin/env python3
"""Thinnest possible programmatic driver for Brogue CE.

Runs the ncurses build (`bin/brogue -t`) as a subprocess whose stdin/stdout are
a pseudo-terminal, so keystrokes can be written in and the rendered screen read
back out. This is the substrate a brogue-gym environment would sit on top of; it
deliberately knows nothing about the game rules.

Requires a build with terminal support:

    make TERMINAL=YES GRAPHICS=YES bin/brogue

`pyte` is optional. Without it you still get raw output and liveness checks;
with it you get a decoded COLS x ROWS character grid.
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import struct
import shutil
import subprocess
import tempfile
import termios
import time
from pathlib import Path

try:
    import pyte
except ImportError:
    pyte = None

# Brogue's fixed screen geometry (src/brogue/Rogue.h).
COLS = 100
ROWS = 34
STAT_BAR_WIDTH = 20            # sidebar columns on the left
MESSAGE_LINES = 3              # message rows along the top
MAP_LEFT = STAT_BAR_WIDTH + 1
MAP_TOP = MESSAGE_LINES
DCOLS = COLS - STAT_BAR_WIDTH - 1
DROWS = ROWS - MESSAGE_LINES - 2

BOOT_SETTLE = 1.0              # seconds of silence that mean "startup finished"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "bin" / "brogue"

_ANSI = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]|\x1b[=>]|\x1b\][^\x07]*\x07")


class BrogueError(RuntimeError):
    pass


class Brogue:
    """A single Brogue process attached to a pty."""

    def __init__(self, seed: int = 1, binary: Path = DEFAULT_BINARY,
                 wizard: bool = False, variant: str | None = None,
                 extra_args: list[str] | None = None):
        self.binary = Path(binary)
        if not self.binary.exists():
            raise BrogueError(
                f"{self.binary} not found -- build it with "
                f"'make TERMINAL=YES GRAPHICS=YES bin/brogue'")

        # Brogue writes its save and recording files into the current directory,
        # and picks a filename with a linear "does this exist yet?" scan. Sharing
        # one directory across runs therefore leaves a pile of LastGame (N) files
        # that makes every subsequent startup slower. Each game gets its own
        # throwaway cwd instead, with --data-dir pointing back at the real assets.
        self.workdir = Path(tempfile.mkdtemp(prefix="brogue-"))

        # -t terminal mode, -n skip the menu, -s fix the seed for reproducibility.
        self.argv = [str(self.binary), "-t", "-n", "-s", str(seed),
                     "--data-dir", str(self.binary.parent)]
        if wizard:
            self.argv.append("--wizard")
        if variant:
            self.argv += ["--variant", variant]
        self.argv += extra_args or []

        self.proc: subprocess.Popen | None = None
        self.fd: int | None = None
        self.raw = bytearray()

        if pyte is not None:
            self._screen = pyte.Screen(COLS, ROWS)
            self._stream = pyte.ByteStream(self._screen)
        else:
            self._screen = self._stream = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "Brogue":
        master, slave = pty.openpty()

        # Size the pty BEFORE the game starts. If the size is set afterwards the
        # child gets a SIGWINCH mid-run, ncurses handles it as a KEY_RESIZE and
        # discards buffered input -- which silently ate the first keystroke on
        # roughly one run in five.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

        env = dict(os.environ, TERM="xterm-256color",
                   COLUMNS=str(COLS), LINES=str(ROWS))
        self.proc = subprocess.Popen(
            self.argv,
            cwd=self.workdir,         # save/recording files land here, not in bin/
            env=env,
            stdin=slave, stdout=slave, stderr=slave,
            close_fds=True,
        )
        os.close(slave)               # the child owns it now
        self.fd = master

        # Startup draws the map and animates the welcome messages, with gaps
        # longer than the normal settle.
        self.drain(settle=BOOT_SETTLE)
        return self

    def stop(self) -> int | None:
        """Terminate the process; return its exit status, or None if not started."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        os.close(self.fd)
        status, self.proc, self.fd = self.proc.returncode, None, None
        shutil.rmtree(self.workdir, ignore_errors=True)
        return status

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- io ----------------------------------------------------------------

    def send(self, keys: str) -> "Brogue":
        os.write(self.fd, keys.encode())
        return self

    def drain(self, settle: float = 0.25, wait: float = 3.0,
              timeout: float = 15.0) -> bytes:
        """Read the game's output.

        Waits up to `wait` seconds for the first byte, then keeps reading until
        the game has been silent for `settle` seconds. Two phases matter because
        Brogue takes a few hundred ms to begin redrawing after a keystroke, so a
        single quiet-gap window either fires too early or wastes time.
        """
        chunk = bytearray()
        deadline = time.monotonic() + timeout
        patience = wait
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.fd], [], [], patience)
            if not ready:
                break
            try:
                data = os.read(self.fd, 65536)
            except OSError:  # child exited, pty hung up
                break
            if not data:
                break
            chunk += data
            patience = settle  # got something; now just wait for it to stop
            if self._stream is not None:
                self._stream.feed(data)
        self.raw += chunk
        return bytes(chunk)

    def step(self, keys: str, **kw) -> bytes:
        """Send keys and read back whatever the game redraws."""
        return self.send(keys).drain(**kw)

    # -- readback ----------------------------------------------------------

    def screen(self) -> list[str]:
        """The decoded ROWS x COLS grid. Requires pyte."""
        if self._screen is None:
            raise BrogueError("pyte is not installed; screen() is unavailable")
        return self._screen.display

    def text(self) -> str:
        """Best-effort text, with or without pyte."""
        if self._screen is not None:
            return "\n".join(self.screen())
        return _ANSI.sub(b"", bytes(self.raw)).decode("utf-8", "replace")

    def find_player(self) -> tuple[int, int] | None:
        """Locate the '@' glyph, in screen coords. Requires pyte.

        Searches the map viewport only -- the sidebar legend also prints "@: You".
        """
        for y in range(MAP_TOP, MAP_TOP + DROWS):
            x = self.screen()[y].find("@", MAP_LEFT)
            if x != -1:
                return x, y
        return None


def smoke_test(seed: int = 1) -> int:
    """Start a game, walk east, prove the process is still alive."""
    print(f"binary : {DEFAULT_BINARY}")
    print(f"pyte   : {'yes' if pyte else 'no (pip install pyte for a screen grid)'}")

    with Brogue(seed=seed) as game:
        print(f"pid    : {game.proc.pid}  argv: {' '.join(game.argv[1:])}")
        print(f"boot   : {len(game.raw)} bytes, alive={game.alive}")

        before = game.find_player() if pyte else None
        out = game.step("l")  # 'l' = move east (vi keys)
        after = game.find_player() if pyte else None

        print(f"sent   : 'l' (move east) -> {len(out)} bytes redrawn")
        if pyte:
            print(f"player : {before} -> {after}")
            print("\n--- screen " + "-" * 60)
            for line in game.screen():
                print(line.rstrip())
            print("-" * 71)
        else:
            tail = game.text().split("\n")[-1][:100]
            print(f"tail   : {tail!r}")

        ok = game.alive
        print(f"\nresult : {'OK - process alive after keystroke' if ok else 'FAILED - process died'}")

    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(smoke_test(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
