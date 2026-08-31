#!/usr/bin/env python3
"""Structured client for Brogue's headless agent mode.

Talks to `bin/brogue --agent`, which skips rendering entirely and exchanges
JSON lines over stdin/stdout. Compared to driving the terminal, this gives
exact numbers instead of a 20-cell health bar, terrain type ids instead of
rendered glyphs, and creature positions without inferring them from the map.

    make TERMINAL=YES GRAPHICS=YES bin/brogue

    with AgentBrogue(seed=42) as game:
        state = game.state
        state = game.step(Action.EAST)
        print(state.hp, state.max_hp, state.depth, state.monsters)
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "bin" / "brogue"
MAX_SEED = 2 ** 31 - 1


class Action(IntEnum):
    """Wire format for actions. Values match agentActionKeys in agent-platform.c."""

    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7
    REST = 8
    SEARCH = 9
    DESCEND = 10
    ASCEND = 11
    PICK_UP = 12
    CONFIRM = 13
    CANCEL = 14
    ACKNOWLEDGE = 15


RAW_KEY_BASE = 1000


def raw_key(ch: str) -> int:
    """Escape hatch: send a literal key the Action table doesn't cover."""
    return RAW_KEY_BASE + ord(ch)


class Prompt(IntEnum):
    """Which modal prompt is waiting for a key. Matches agentPromptKinds."""

    NONE = 0
    CONFIRM = 1        # a yes/no box: "Dive into the depths?"
    ACKNOWLEDGE = 2    # --MORE--
    TEXT = 3           # a string is being typed


class BrogueAgentError(RuntimeError):
    pass


@dataclass
class Monster:
    x: int
    y: int
    kind: int
    hp: int
    max_hp: int
    state: int
    visible: bool
    name: str = ""


@dataclass
class AgentState:
    """One observation. Coordinates are dungeon cells, 79x29."""

    turn: int
    depth: int
    gold: int
    strength: int
    hp: int
    max_hp: int
    x: int
    y: int
    terrain: list[list[int]]      # tile type id per cell
    visibility: list[list[int]]   # bit 0 discovered, bit 1 visible now
    monsters: list[Monster]
    dead: bool
    game_over: bool
    text_input: bool
    killed_by: str = ""          # empty until the run ends
    killed_by_custom: bool = False   # False: a monster name. True: a whole phrase.
    prompt: int = 0              # a Prompt: what the game is waiting to be told

    @property
    def player(self) -> tuple[int, int]:
        return self.x, self.y

    def discovered(self, x: int, y: int) -> bool:
        return bool(self.visibility[y][x] & 1)

    def visible(self, x: int, y: int) -> bool:
        return bool(self.visibility[y][x] & 2)


@dataclass
class Legend:
    """Sent once per process: what the integers in each state mean."""

    width: int
    height: int
    action_count: int
    raw_key_base: int
    tiles: list[str]
    monsters: list[str]
    tile_flags: list[int]        # terrainFlagCatalog bits, per tile type
    tile_mech_flags: list[int]   # terrainMechanicalFlagCatalog bits, per tile type

    def tile_name(self, tile_id: int) -> str:
        return self.tiles[tile_id] if 0 <= tile_id < len(self.tiles) else f"?{tile_id}"

    def monster_name(self, kind: int) -> str:
        return self.monsters[kind] if 0 <= kind < len(self.monsters) else f"?{kind}"


class AgentBrogue:
    """A Brogue process in headless agent mode."""

    def __init__(self, seed: int | None = None, binary: Path = DEFAULT_BINARY,
                 wizard: bool = False, variant: str | None = None,
                 rng: random.Random | None = None):
        """
        seed=<int>  every episode replays that dungeon
        seed=None   every episode draws a fresh one
        """
        self.binary = Path(binary)
        if not self.binary.exists():
            raise BrogueAgentError(
                f"{self.binary} not found -- build it with "
                f"'make TERMINAL=YES GRAPHICS=YES bin/brogue'")

        self.fixed_seed = seed
        self.rng = rng or random.Random()
        self.wizard = wizard
        self.variant = variant

        self.seed: int | None = None
        self.proc: subprocess.Popen | None = None
        self.workdir: Path | None = None
        self.legend: Legend | None = None
        self.state: AgentState | None = None

    # -- lifecycle ---------------------------------------------------------

    def _next_seed(self, seed: int | None) -> int:
        if seed is not None:
            return seed
        if self.fixed_seed is not None:
            return self.fixed_seed
        return self.rng.randrange(1, MAX_SEED)

    def start(self, seed: int | None = None) -> AgentState:
        if self.proc is not None:
            raise BrogueAgentError("already started; call reset() instead")

        self.seed = self._next_seed(seed)
        # Brogue writes save and recording files into the working directory and
        # names them with a linear existence scan, so give each episode its own.
        self.workdir = Path(tempfile.mkdtemp(prefix="brogue-agent-"))

        argv = [str(self.binary), "--agent", "-n", "-s", str(self.seed),
                "--data-dir", str(self.binary.parent)]
        if self.wizard:
            argv.append("--wizard")
        if self.variant:
            argv += ["--variant", self.variant]

        self.proc = subprocess.Popen(
            argv, cwd=self.workdir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        message = self._read_message()
        if message is None or message.get("type") != "legend":
            raise BrogueAgentError(f"expected a legend first, got {message!r}")
        self.legend = Legend(
            width=message["width"], height=message["height"],
            action_count=message["action_count"], raw_key_base=message["raw_key_base"],
            tiles=message["tiles"], monsters=message["monsters"],
            tile_flags=message["tile_flags"],
            tile_mech_flags=message["tile_mech_flags"],
        )

        self.state = self._read_state()
        return self.state

    def stop(self) -> int | None:
        if self.proc is None:
            return None
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        status, self.proc = self.proc.returncode, None
        shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir = None
        return status

    def reset(self, seed: int | None = None) -> AgentState:
        """End this episode and start a fresh one. Returns the first state."""
        self.stop()
        return self.start(seed)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- protocol ----------------------------------------------------------

    def _read_message(self) -> dict | None:
        """Next JSON object, skipping any plain text the game prints."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None                     # the game exited
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def _read_state(self) -> AgentState:
        message = self._read_message()
        if message is None:
            if self.state is not None:
                self.state.game_over = True     # process exited; episode is over
                return self.state
            raise BrogueAgentError("game exited before sending any state")
        if message.get("type") != "state":
            raise BrogueAgentError(f"expected a state, got {message.get('type')!r}")

        names = self.legend.monster_name if self.legend else (lambda k: "")
        monsters = [
            Monster(x=m["x"], y=m["y"], kind=m["kind"], hp=m["hp"],
                    max_hp=m["max_hp"], state=m["state"], visible=m["visible"],
                    name=names(m["kind"]))
            for m in message["monsters"]
        ]
        return AgentState(
            turn=message["turn"], depth=message["depth"], gold=message["gold"],
            strength=message["strength"], hp=message["hp"], max_hp=message["max_hp"],
            x=message["player"]["x"], y=message["player"]["y"],
            terrain=message["terrain"], visibility=message["visibility"],
            monsters=monsters, dead=message["dead"],
            game_over=message["game_over"], text_input=message["text_input"],
            killed_by=message.get("killed_by", ""),
            killed_by_custom=message.get("killed_by_custom", False),
            prompt=message.get("prompt", 0),
        )

    def step(self, action: int) -> AgentState:
        """Send one action and return the resulting state."""
        if self.proc is None:
            raise BrogueAgentError("not started")
        if self.state is not None and self.state.game_over:
            return self.state
        try:
            self.proc.stdin.write(f"{int(action)}\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.state.game_over = True
            return self.state
        self.state = self._read_state()
        return self.state


def smoke_test(seed: int = 42) -> int:
    with AgentBrogue(seed=seed) as game:
        s = game.state
        print(f"seed   : {game.seed}")
        print(f"legend : {game.legend.width}x{game.legend.height} grid, "
              f"{len(game.legend.tiles)} tile types, {len(game.legend.monsters)} monster types")
        print(f"start  : turn={s.turn} depth={s.depth} hp={s.hp}/{s.max_hp} "
              f"str={s.strength} at {s.player}")

        s = game.step(Action.EAST)
        print(f"east   : turn={s.turn} at {s.player}")
        here = s.terrain[s.y][s.x]
        print(f"tile   : {here} = {game.legend.tile_name(here)!r}")
        print(f"seen   : {sum(v & 1 != 0 for row in s.visibility for v in row)} cells discovered")
        print(f"monsters: {[(m.name, m.x, m.y, f'{m.hp}/{m.max_hp}') for m in s.monsters][:5]}")
        ok = s.x == game.state.x and not s.game_over
        print(f"\nresult : {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(smoke_test(int(sys.argv[1]) if len(sys.argv) > 1 else 42))
