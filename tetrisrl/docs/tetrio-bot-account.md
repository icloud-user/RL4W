# Playing the bot on TETR.IO (and why there is no shortcut)

Researched 2026-09 for the versus mode. Sources are listed at the bottom; the
short version is that **there is no public bot API, and the only sanctioned way to
play a bot on TETR.IO is a staff-granted bot account in a custom room.**

## What the rules actually say

TETR.IO's own API page is explicit:

> This document **only** describes the TETRA CHANNEL API (at https://ch.tetr.io/api/),
> not the main game API (at https://tetr.io/api/). Usage of the main game API is
> **NOT ALLOWED** without explicit, written consent. Unauthorized usage of the main
> game API may result in permanent suspension of your account.

and the Community Rules, rule 2:

> DO NOT (ATTEMPT TO) CHEAT IN ANY WAY. Using any third-party utilities, loopholes
> or exploits of any kind to get any sort of advantage is not OK. This includes but
> is not limited to things like macro programs, timescale programs, **bots**, replay
> forgery, lagswitches, assistance tools (like solution finders) and so on.

Rule 3 adds a strict one-account-per-user policy and forbids one account being used
by more than one person. Practical consequences:

* **Do not** point this bot at your own account. Protocol emulation on a normal
  account is an explicit termination offence ("bots that are not verified will be
  destroyed on sight, together with their creator" — patch notes ALPHA 3.2.4).
* **Do not** use a human as a relay either. Rule 2 covers assistance tools, and
  rule 3 requires not misrepresenting your skill.
* The read-only TETRA CHANNEL API (`ch.tetr.io/api`) is fine and needs no account,
  but it cannot play a game.

## The sanctioned path: ask for a bot account

Announced in patch notes ALPHA 3.2.4 (2020-06-16):

> Bot accounts are now supported. If you wish to create a simple bot, send me a DM
> or email with an outline of the bot's functionality.

1. Write a short outline of what this bot does, who runs it, and that it will play
   **only in custom rooms** — bots cannot enter TETRA LEAGUE, Quick Play or solo
   modes.
2. Send it to osk (contact details at https://osk.sh/, or TETR.IO support). It is a
   manual, discretionary review with **no published turnaround time**; there is no
   form and no queue you can check.
3. If approved you get a bot account (`role: "bot"`, with a `botmaster` field
   visible on the public profile) and a token.
4. Implement the client side of the **Ribbon** WebSocket protocol. It is
   undocumented and changes without notice — the patch notes carry explicit
   "note for bot developers" entries (ALPHA 5.1.1, 5.1.3, 6.0.3, 6.3.0). Existing
   clients: `tetr.js` (TypeScript), `@haelp/teto`/Triangle.js (TypeScript, has a
   gameplay-bot quickstart), `tetry` (Python, last released 2021 and very likely
   stale after later protocol breaks).
5. Run the bot from that account, host a custom room, and have a human join.

The gap between step 3 and step 5 is the real work: **this repository does not
implement Ribbon.** Everything here is engine, rules and decisions; a TETR.IO
client would sit on top of `tetrisrl.versus` and translate game events into
`Battle.after_lock(...)` calls and back.

## What to use instead, today

* `python -m tetrisrl battle` — the local versus mode, which implements TETR.IO's
  attack table, combo multiplier, B2B/Surge, oldest-first cancelling, the 20-frame
  garbage queue with a cap of 8, tanking only on non-clearing locks, and
  configurable clean/messy hole generation.
* `python tools/versus_match.py` — headless matches, for measuring the difficulty
  presets without playing them.

If you would like TETR.IO itself, the honest summary is: apply for the bot account
first, and only then is there something to build against.

## Sources

* https://tetr.io/about/api/ — the "main game API is NOT ALLOWED" notice
* https://tetr.io/about/rules/ — rules 2 and 3, quoted above
* https://tetr.io/about/patchnotes/notes.json — ALPHA 3.2.4 (bot accounts), and the
  bot-developer notes in 5.1.1, 5.1.3, 6.0.3, 6.3.0
* https://github.com/tetrio/issues/issues/539 — external bots / local multiplayer,
  still open; #1142 and #1077 closed as not planned
* https://github.com/rumia-moe/tetrio-bot-docs — reverse-engineered bot protocol
  notes (community, written against v6.0.4)
* https://github.com/tetrjs/tetr.js — "the token provided must match an account
  that has been approved as a bot account"
* https://github.com/halp1/triangle and https://triangle.haelp.dev — the engine
  documentation the versus numbers in `docs/tetrio-versus-ruleset.md` come from
