#!/usr/bin/env python3
"""Thinnest possible programmatic driver for Brogue CE.

Runs the ncurses build (`bin/brogue -t`) as a subprocess whose stdin/stdout are
a pseudo-terminal, so keystrokes can be written in and the rendered screen read
back out. This is the substrate a brogue-gym environment sits on top of; it
knows about the terminal, not about game rules.

Requires a build with terminal support:

    make TERMINAL=YES GRAPHICS=YES bin/brogue

`pyte` is optional for raw driving, but required for parsed state.
"""

from __future__ import annotations

import fcntl
import os
import pty
import random
import re
import select
import shutil
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

try:
    from brogue_state import COLS, ROWS, GameState, parse
except ImportError:  # importable from outside python/ too
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from brogue_state import COLS, ROWS, GameState, parse

try:
    import pyte
except ImportError:
    pyte = None

BOOT_SETTLE = 1.0              # seconds of silence that mean "startup finished"
MAX_SEED = 2 ** 31 - 1

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "bin" / "brogue"

_ANSI = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]|\x1b[=>]|\x1b\][^\x07]*\x07")

# vi-keys, which is what the game reads for movement.
MOVES = {"north": "k", "south": "j", "west": "h", "east": "l",
         "northwest": "y", "northeast": "u",
         "southwest": "b", "southeast": "n"}


class BrogueError(RuntimeError):
    pass


class Brogue:
    """A single Brogue process attached to a pty."""

    def __init__(self, seed: int | None = None, binary: Path = DEFAULT_BINARY,
                 wizard: bool = False, variant: str | None = None,
                 extra_args: list[str] | None = None,
                 rng: random.Random | None = None):
        """
        seed=<int>  every episode replays that dungeon (debugging, reproducibility)
        seed=None   every episode draws a fresh random dungeon (generalization)

        `reset(seed=...)` overrides it for a single episode, and the seed
        actually in play is always readable as `.seed`.
        """
        self.binary = Path(binary)
        if not self.binary.exists():
            raise BrogueError(
                f"{self.binary} not found -- build it with "
                f"'make TERMINAL=YES GRAPHICS=YES bin/brogue'")

        self.fixed_seed = seed
        self.rng = rng or random.Random()
        self.wizard = wizard
        self.variant = variant
        self.extra_args = extra_args or []

        self.seed: int | None = None
        self.proc: subprocess.Popen | None = None
        self.fd: int | None = None
        self.workdir: Path | None = None
        self.raw = bytearray()
        self._screen = self._stream = None

    # -- lifecycle ---------------------------------------------------------

    def _next_seed(self, seed: int | None) -> int:
        if seed is not None:
            return seed
        if self.fixed_seed is not None:
            return self.fixed_seed
        return self.rng.randrange(1, MAX_SEED)

    def start(self, seed: int | None = None) -> "Brogue":
        if self.proc is not None:
            raise BrogueError("already started; call reset() instead")

        self.seed = self._next_seed(seed)
        self.raw = bytearray()
        if pyte is not None:
            self._screen = pyte.Screen(COLS, ROWS)
            self._stream = pyte.ByteStream(self._screen)

        argv = [str(self.binary), "-t", "-n", "-s", str(self.seed),
                "--data-dir", str(self.binary.parent)]
        if self.wizard:
            argv.append("--wizard")
        if self.variant:
            argv += ["--variant", self.variant]
        argv += self.extra_args
        self.argv = argv

        # Brogue writes its save and recording files into the current directory,
        # and picks a filename with a linear "does this exist yet?" scan. Sharing
        # one directory across episodes leaves a pile of LastGame (N) files that
        # makes every later startup slower, so each game gets a throwaway cwd
        # with --data-dir pointing back at the real assets.
        self.workdir = Path(tempfile.mkdtemp(prefix="brogue-"))

        master, slave = pty.openpty()
        # Size the pty before the game starts, so ncurses sees 100x34 from its
        # first initscr() and never has to handle a resize.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

        self.proc = subprocess.Popen(
            argv,
            cwd=self.workdir,
            env=dict(os.environ, TERM="xterm-256color",
                     COLUMNS=str(COLS), LINES=str(ROWS)),
            stdin=slave, stdout=slave, stderr=slave,
            close_fds=True,
        )
        os.close(slave)
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
        self.workdir = None
        return status

    def reset(self, seed: int | None = None) -> GameState:
        """End the current episode and begin a fresh one. Returns the first state."""
        self.stop()
        self.start(seed)
        return self.state()

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
        Brogue takes a couple hundred ms to begin redrawing after a keystroke, so
        a single quiet-gap window either fires too early or wastes time.
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

    def step(self, keys: str, **kw) -> GameState:
        """Send keys, read the redraw, and return the parsed state."""
        self.send(keys).drain(**kw)
        return self.state()

    def move(self, direction: str, **kw) -> GameState:
        """Move one step: 'north', 'southeast', ... See MOVES."""
        try:
            key = MOVES[direction]
        except KeyError:
            raise BrogueError(
                f"unknown direction {direction!r}; expected one of {sorted(MOVES)}")
        return self.step(key, **kw)

    # -- readback ----------------------------------------------------------

    def screen(self) -> list[str]:
        """The decoded ROWS x COLS grid, sidebar included. Requires pyte."""
        if self._screen is None:
            raise BrogueError("pyte is not installed; screen() is unavailable")
        return self._screen.display

    def state(self) -> GameState:
        """The current screen, parsed into structured state. Requires pyte."""
        if self._screen is None:
            raise BrogueError("pyte is not installed; state() is unavailable")
        state = parse(self._screen)
        if not self.alive:
            state.game_over = True
        return state

    def text(self) -> str:
        """Best-effort text, with or without pyte."""
        if self._screen is not None:
            return "\n".join(self.screen())
        return _ANSI.sub(b"", bytes(self.raw)).decode("utf-8", "replace")

    def find_player(self) -> tuple[int, int] | None:
        """Player position in map coordinates, or None if off-screen."""
        return self.state().player


def smoke_test(seed: int = 1) -> int:
    """Start a game, walk east, and report the parsed state."""
    print(f"binary : {DEFAULT_BINARY}")
    print(f"pyte   : {'yes' if pyte else 'no (pip install pyte)'}")

    with Brogue(seed=seed) as game:
        print(f"pid    : {game.proc.pid}   seed: {game.seed}")
        print(f"boot   : {len(game.raw)} bytes, alive={game.alive}")

        if not pyte:
            print(f"tail   : {game.text().splitlines()[-1][:90]!r}")
            print(f"\nresult : {'OK' if game.alive else 'FAILED'}")
            return 0 if game.alive else 1

        before = game.state()
        after = game.move("east")

        print(f"player : {before.player} -> {after.player}")
        print(f"depth  : {after.depth}   hp: {after.hp_fraction:.2f}   "
              f"nutrition: {after.nutrition_fraction:.2f}")
        print(f"str    : {after.strength}   armor: {after.armor}   gold: {after.gold}")
        print(f"msgs   : {after.messages}")
        print(f"dead   : {after.dead}")
        print("\n--- map " + "-" * 63)
        print(after.render())
        print("-" * 71)

        ok = game.alive and after.player == (before.player[0] + 1, before.player[1])
        print(f"\nresult : {'OK - moved one tile east' if ok else 'FAILED'}")

    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(smoke_test(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
