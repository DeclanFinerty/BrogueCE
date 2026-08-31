/*
 *  agent-platform.c
 *
 *  A headless backend for programmatic play. It implements the same
 *  brogueConsole interface as the SDL and curses backends, but instead of
 *  drawing a screen and reading a keyboard it writes the game state to stdout
 *  as JSON and reads an action number from stdin.
 *
 *  Rendering is skipped entirely: plotChar does nothing and
 *  pauseForMilliseconds returns immediately, so there are no animation delays.
 *
 *  Protocol, one JSON object per line in each direction:
 *
 *    out  {"type":"legend",...}   once, before the first state
 *    out  {"type":"state",...}    whenever the game wants input
 *    in   <integer>\n             the action to take
 *
 *  Actions 0-7 are the eight movement directions, clockwise from north. The
 *  rest are listed in agentActionKeys below. An action of AGENT_RAW_KEY_BASE
 *  or more sends the raw key code (action - AGENT_RAW_KEY_BASE), which is the
 *  escape hatch for prompts and menus this table doesn't cover.
 *
 *  The process exits when the game ends or when stdin closes. Starting a new
 *  episode means starting a new process.
 */

#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "GlobalsBase.h"
#include "Globals.h"

#define AGENT_RAW_KEY_BASE 1000

// Movement is clockwise from north, then the handful of non-movement actions
// an early agent needs. Indices are the wire format, so only ever append.
static const int agentActionKeys[] = {
    'k', 'u', 'l', 'n', 'j', 'b', 'h', 'y',     // N NE E SE S SW W NW
    REST_KEY,                                   // 8
    SEARCH_KEY,                                 // 9
    DESCEND_KEY,                                // 10
    ASCEND_KEY,                                 // 11
    ',',                                        // 12  pick up
    RETURN_KEY,                                 // 13  confirm
    ESCAPE_KEY,                                 // 14  cancel
    ACKNOWLEDGE_KEY,                            // 15  dismiss a message
};

#define AGENT_ACTION_COUNT ((int) (sizeof(agentActionKeys) / sizeof(agentActionKeys[0])))

static boolean legendSent = false;

/*
 * True once the run is finished. gameOver() sets MB_IS_DYING and then loops on
 * nextBrogueEvent until it is acknowledged, long before rogue.gameHasEnded is
 * set, so checking only gameHasEnded leaves an agent trapped in the death
 * screen answering prompts forever.
 */
static boolean agentEpisodeOver() {
    return rogue.gameHasEnded
        || rogue.quit
        || (player.bookkeepingFlags & MB_IS_DYING) != 0;
}

static void agentJsonString(const char *s) {
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char) *s;
        if (c == '"' || c == '\\') {
            putchar('\\');
            putchar(c);
        } else if (c < 0x20) {
            printf("\\u%04x", c);
        } else {
            putchar(c);
        }
    }
    putchar('"');
}

// Sent once, so the other end can turn the integers in each state into names.
static void agentEmitLegend() {
    printf("{\"type\":\"legend\",\"width\":%i,\"height\":%i", DCOLS, DROWS);
    printf(",\"action_count\":%i,\"raw_key_base\":%i", AGENT_ACTION_COUNT, AGENT_RAW_KEY_BASE);

    printf(",\"tiles\":[");
    for (int i = 0; i < NUMBER_TILETYPES; i++) {
        if (i) putchar(',');
        agentJsonString(tileCatalog[i].description);
    }

    // Terrain flags per tile type (terrainFlagCatalog / terrainMechanicalFlagCatalog),
    // so the agent decides passability and danger from the same bits the game uses
    // instead of guessing from the description strings.
    printf("],\"tile_flags\":[");
    for (int i = 0; i < NUMBER_TILETYPES; i++) {
        if (i) putchar(',');
        printf("%lu", (unsigned long) tileCatalog[i].flags);
    }

    printf("],\"tile_mech_flags\":[");
    for (int i = 0; i < NUMBER_TILETYPES; i++) {
        if (i) putchar(',');
        printf("%lu", (unsigned long) tileCatalog[i].mechFlags);
    }

    printf("],\"monsters\":[");
    for (int i = 0; i < NUMBER_MONSTER_KINDS; i++) {
        if (i) putchar(',');
        agentJsonString(monsterCatalog[i].monsterName);
    }

    printf("]}\n");
}

static void agentEmitGrids() {
    // The tile type of the highest-priority layer, which is what the player
    // would effectively be standing in: a wall, or lava over the floor, or
    // grass over the floor.
    printf(",\"terrain\":[");
    for (int y = 0; y < DROWS; y++) {
        if (y) putchar(',');
        putchar('[');
        for (int x = 0; x < DCOLS; x++) {
            if (x) putchar(',');
            printf("%i", (int) pmap[x][y].layers[highestPriorityLayer(x, y, false)]);
        }
        putchar(']');
    }

    // Bit 0: the player has discovered this cell. Bit 1: it is visible now.
    printf("],\"visibility\":[");
    for (int y = 0; y < DROWS; y++) {
        if (y) putchar(',');
        putchar('[');
        for (int x = 0; x < DCOLS; x++) {
            int v = 0;
            if (pmap[x][y].flags & DISCOVERED) v |= 1;
            if (pmap[x][y].flags & ANY_KIND_OF_VISIBLE) v |= 2;
            if (x) putchar(',');
            printf("%i", v);
        }
        putchar(']');
    }
    putchar(']');
}

static void agentEmitMonsters() {
    printf(",\"monsters\":[");
    if (monsters != NULL) {
        boolean first = true;
        for (creatureIterator it = iterateCreatures(monsters); hasNextCreature(it);) {
            creature *monst = nextCreature(&it);
            if (!first) putchar(',');
            first = false;
            printf("{\"x\":%i,\"y\":%i,\"kind\":%i,\"hp\":%i,\"max_hp\":%i,\"state\":%i,\"visible\":%s}",
                   monst->loc.x, monst->loc.y, (int) monst->info.monsterID,
                   monst->currentHP, monst->info.maxHP, (int) monst->creatureState,
                   canSeeMonster(monst) ? "true" : "false");
        }
    }
    putchar(']');
}

static void agentEmitState(boolean textInput) {
    printf("{\"type\":\"state\",\"turn\":%lu,\"depth\":%i,\"gold\":%lu,\"strength\":%i",
           (unsigned long) rogue.playerTurnNumber, rogue.depthLevel,
           (unsigned long) rogue.gold, rogue.strength);
    printf(",\"hp\":%i,\"max_hp\":%i", player.currentHP, player.info.maxHP);
    printf(",\"player\":{\"x\":%i,\"y\":%i}", player.loc.x, player.loc.y);
    printf(",\"text_input\":%s", textInput ? "true" : "false");

    // Which prompt is blocking, so the agent can answer it on purpose. The
    // dangerous moves the game asks about -- diving into a chasm, walking into
    // caustic gas -- are legitimate plays, so agent mode reports the question
    // rather than answering it.
    // A confirmation box runs through the button loop, which asks for textInput,
    // so the specific kind wins over the generic flag.
    printf(",\"prompt\":%i", agentPromptKind != AGENT_PROMPT_NONE ? agentPromptKind
                              : (textInput ? AGENT_PROMPT_TEXT : AGENT_PROMPT_NONE));
    printf(",\"dead\":%s", player.currentHP <= 0 ? "true" : "false");
    printf(",\"game_over\":%s", agentEpisodeOver() ? "true" : "false");

    // Why the run ended. Empty until gameOver() runs; killed_by_custom false
    // means the string is a monster name, true means it is a whole phrase.
    printf(",\"killed_by\":");
    agentJsonString(agentKilledBy);
    printf(",\"killed_by_custom\":%s", agentKilledByCustom ? "true" : "false");
    agentEmitGrids();
    agentEmitMonsters();
    printf("}\n");
    fflush(stdout);
}

static void agent_gameLoop() {
    exit(rogueMain());
}

static boolean agent_pauseForMilliseconds(short milliseconds, PauseBehavior behavior) {
    return false;   // never wait; there is nothing to animate
}

static void agent_nextKeyOrMouseEvent(rogueEvent *returnEvent, boolean textInput, boolean colorsDance) {
    char line[64];
    int action, key;

    if (!legendSent) {
        agentEmitLegend();
        legendSent = true;
    }
    agentEmitState(textInput);

    if (agentEpisodeOver()) {
        // The episode is over. The caller starts a new process for the next one.
        exit(EXIT_STATUS_SUCCESS);
    }

    if (!fgets(line, sizeof(line), stdin)) {
        exit(EXIT_STATUS_SUCCESS);  // stdin closed
    }
    action = atoi(line);

    if (action >= AGENT_RAW_KEY_BASE) {
        key = action - AGENT_RAW_KEY_BASE;
    } else if (action >= 0 && action < AGENT_ACTION_COUNT) {
        key = agentActionKeys[action];
    } else {
        key = ACKNOWLEDGE_KEY;
    }

    returnEvent->eventType = KEYSTROKE;
    returnEvent->param1 = key;
    returnEvent->param2 = 0;
    returnEvent->controlKey = 0;
    returnEvent->shiftKey = (key >= 'A' && key <= 'Z');
}

static void agent_plotChar(enum displayGlyph ch, short x, short y,
                           short foreRed, short foreGreen, short foreBlue,
                           short backRed, short backGreen, short backBlue) {
    return;   // headless
}

static boolean agent_modifierHeld(int modifier) {
    return false;
}

struct brogueConsole agentConsole = {
    agent_gameLoop,
    agent_pauseForMilliseconds,
    agent_nextKeyOrMouseEvent,
    agent_plotChar,
    NULL,
    agent_modifierHeld,
    NULL,
    NULL,
    NULL
};
