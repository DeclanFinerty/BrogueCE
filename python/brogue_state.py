#!/usr/bin/env python3
"""Turn a rendered Brogue screen into structured state.

Deliberately partial. It extracts what a first learning loop needs -- the map
grid, where the player is, health and depth -- and leaves the rest of the UI
alone. See `docs/programmatic-control.md` for what the screen can and cannot
tell us.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Screen geometry, from src/brogue/Rogue.h.
COLS = 100
ROWS = 34
STAT_BAR_WIDTH = 20
MESSAGE_LINES = 3
MAP_LEFT = STAT_BAR_WIDTH + 1        # column 20 is the sidebar/map separator
MAP_TOP = MESSAGE_LINES
DCOLS = COLS - STAT_BAR_WIDTH - 1    # 79
DROWS = ROWS - MESSAGE_LINES - 2     # 29
DEPTH_ROW = ROWS - 1                 # the bar along the very bottom
FLAVOR_ROW = ROWS - 2                # one-line terrain description

_DEPTH = re.compile(r"Depth:\s*(\d+)")
_STRENGTH = re.compile(r"Str:\s*(\d+)")
_ARMOR = re.compile(r"Armor:\s*(\d+)")
_GOLD = re.compile(r"Gold:\s*(\d+)")
_DEATH = re.compile(r"You die\.\.\.|Killed by", re.I)
# The run-summary screen the game shows once an episode is over, whether the
# player died, quit, or escaped with the amulet.
_GAME_OVER = re.compile(r"-- HIGH SCORES --|Press space to continue|"
                        r"Save recording as|Killed by .* on depth \d+", re.I)
# The recording-save prompt spells out how the run ended.
_OUTCOME = re.compile(r"#\d+ (.+?)\.broguerec")


@dataclass
class GameState:
    """One observation, parsed from a single screen."""

    grid: list[str]                              # DROWS rows of DCOLS map chars
    fg: list[list[str]]                          # per-cell foreground, 'rrggbb'
    bg: list[list[str]]                          # per-cell background, 'rrggbb'
    player: tuple[int, int] | None               # (col, row) in map coordinates
    hp_fraction: float | None                    # 0.0-1.0, see note below
    nutrition_fraction: float | None
    depth: int | None
    gold: int | None
    strength: int | None
    armor: int | None
    messages: list[str] = field(default_factory=list)
    dead: bool = False          # the player's health reached zero
    game_over: bool = False     # the episode has ended, for any reason
    outcome: str | None = None  # e.g. "Killed by a rat on depth 3", when shown

    def tile(self, x: int, y: int) -> str:
        return self.grid[y][x]

    def render(self) -> str:
        return "\n".join(self.grid)


def _luminance(hex_color: str) -> float:
    """Rough perceived brightness of a 'rrggbb' string, 0.0-1.0."""
    if not isinstance(hex_color, str) or len(hex_color) != 6:
        return 0.0
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return 0.0
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _bar_fraction(cells) -> float | None:
    """Read a 20-cell progress bar as a fraction.

    printProgressBar() in src/brogue/IO.c paints the filled part of the bar in
    the bar colour and the rest in the same colour darkened by 75%, so the split
    shows up as a step in background brightness. There is no number on screen,
    which caps the resolution at 1/20 -- the one cell on the boundary is blended
    to encode the remainder, but that precision is not worth reconstructing.
    """
    lums = [_luminance(c.bg) for c in cells]
    if not lums:
        return None
    lo, hi = min(lums), max(lums)
    if hi - lo < 0.02:          # uniform bar: completely full (or completely empty)
        return 1.0
    midpoint = (lo + hi) / 2
    return sum(1 for l in lums if l >= midpoint) / len(lums)


def _sidebar_rows(screen) -> list[tuple[int, str, list]]:
    rows = []
    for y in range(ROWS):
        cells = [screen.buffer[y][x] for x in range(STAT_BAR_WIDTH)]
        rows.append((y, "".join(c.data for c in cells), cells))
    return rows


def parse(screen) -> GameState:
    """Build a GameState from a pyte Screen."""
    buf = screen.buffer

    grid, fg, bg = [], [], []
    for y in range(MAP_TOP, MAP_TOP + DROWS):
        row = [buf[y][x] for x in range(MAP_LEFT, COLS)]
        grid.append("".join(c.data for c in row))
        fg.append([c.fg for c in row])
        bg.append([c.bg for c in row])

    player = None
    for y, line in enumerate(grid):
        x = line.find("@")
        if x != -1:
            player = (x, y)
            break

    sidebar = _sidebar_rows(screen)
    sidebar_text = "\n".join(text for _, text, _ in sidebar)

    hp = nutrition = None
    dead = False
    for _, text, cells in sidebar:
        stripped = text.strip()
        if "Health" in stripped:
            hp = _bar_fraction(cells)
        elif stripped == "Dead":
            hp, dead = 0.0, True
        elif "Nutrition" in stripped:
            nutrition = _bar_fraction(cells)

    messages = []
    for y in range(MESSAGE_LINES):
        line = "".join(buf[y][x].data for x in range(MAP_LEFT, COLS)).strip()
        if line:
            messages.append(line)

    bottom = "".join(buf[DEPTH_ROW][x].data for x in range(COLS))
    flavor = "".join(buf[FLAVOR_ROW][x].data for x in range(COLS)).strip()

    def _first(pattern, *sources):
        for src in sources:
            m = pattern.search(src)
            if m:
                return int(m.group(1))
        return None

    full_screen = "\n".join(screen.display)
    game_over = bool(_GAME_OVER.search(full_screen))
    if _DEATH.search(full_screen):
        dead = game_over = True

    outcome_match = _OUTCOME.search(full_screen)
    outcome = outcome_match.group(1).strip() if outcome_match else None
    if outcome and re.match(r"Killed by", outcome, re.I):
        dead = True

    return GameState(
        grid=grid, fg=fg, bg=bg, player=player,
        hp_fraction=hp, nutrition_fraction=nutrition,
        depth=_first(_DEPTH, bottom, sidebar_text),
        gold=_first(_GOLD, sidebar_text) or 0,
        strength=_first(_STRENGTH, sidebar_text),
        armor=_first(_ARMOR, sidebar_text),
        messages=messages, dead=dead, game_over=game_over, outcome=outcome,
    )
