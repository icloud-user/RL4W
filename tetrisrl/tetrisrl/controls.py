"""Player input settings: bindings, DAS/ARR, SDF and lock delay.

Everything a player can rebind lives here, in the units TETR.IO players already
know -- milliseconds for DAS, ARR and lock delay, and a soft-drop *factor* (a
multiplier of gravity, where 0 means instant) -- converted once into the seconds
the renderer works in.

Two files, on purpose:

* ``config.yaml``'s ``input:`` section is the project default, documented and
  commented like everything else there.
* ``settings.yaml`` sits next to it and is what the in-game settings screen
  writes. It is merged *over* the config, so a player's own bindings survive both
  a config edit and a ``git pull`` -- and the settings screen never has to rewrite
  the commented config file, which ``yaml.safe_dump`` would strip bare.

Every action accepts **two** keys. That is the ``list`` in the config: any key in
the list does the job, and the settings screen edits slot 1 and slot 2 separately.
"""
from __future__ import annotations

import os

import pygame
import yaml

#: Every bindable action: ``(name, label, group)``. The order is the order the
#: settings screen shows, grouped so movement sits together.
ACTIONS = (
    ('move_left', 'Move left', 'movement'),
    ('move_right', 'Move right', 'movement'),
    ('soft_drop', 'Soft drop', 'movement'),
    ('hard_drop', 'Hard drop', 'movement'),
    ('rotate_cw', 'Rotate right', 'rotation'),
    ('rotate_ccw', 'Rotate left', 'rotation'),
    ('rotate_180', 'Rotate 180', 'rotation'),
    ('hold', 'Hold', 'pieces'),
    ('pause', 'Pause', 'game'),
    ('restart', 'Restart', 'game'),
    ('new_seed', 'New seed', 'game'),
    ('quit', 'Quit', 'game'),
)
ACTION_NAMES = tuple(name for name, _label, _group in ACTIONS)
ACTION_LABELS = {name: label for name, label, _group in ACTIONS}
#: Actions the game loop owns rather than the piece input handler.
LOOP_ACTIONS = ('pause', 'restart', 'new_seed', 'quit')

#: Defaults follow TETR.IO: arrows to move, Z/X or Ctrl/Up to rotate, A for 180,
#: C or Shift to hold. Three actions ship double-bound so the mechanism is
#: visible in the shipped config rather than only in the docs.
DEFAULT_BINDINGS = {
    'move_left': ['left'],
    'move_right': ['right'],
    'soft_drop': ['down'],
    'hard_drop': ['space'],
    'rotate_cw': ['up', 'x'],
    'rotate_ccw': ['z', 'left ctrl'],
    'rotate_180': ['a'],
    'hold': ['c', 'left shift'],
    'pause': ['p'],
    'restart': ['r'],
    'new_seed': ['n'],
    'quit': ['escape'],
}

#: DAS 133 ms and ARR 0 (instant) are the TETR.IO defaults people expect; SDF 41
#: is a typical soft-drop factor, and 500 ms of lock delay is the guideline value
#: this project already used.
DEFAULT_TIMING = {
    'das_ms': 133,
    'arr_ms': 0,
    'sdf': 41.0,
    #: Soft drop that teleports to the landing row instead of being fast. A flag
    #: rather than "SDF 0" because it is a different *kind* of thing, not a value
    #: on the same dial -- and because reaching 0 by nudging from 41 is 41 presses.
    'soft_drop_instant': False,
    'lock_delay_ms': 500,
    'lock_reset_limit': 15,
    #: DAS Cut Delay: how long horizontal auto-repeat is frozen after a rotation
    #: or a new piece. See :attr:`InputSettings.dcd`. Off at 0, which is the
    #: default because it changes the feel of movement rather than fixing it.
    'dcd_ms': 0,
}

SETTINGS_NAME = 'settings.yaml'
MAX_BINDINGS = 2
#: Floor on the soft-drop interval. It exists only to keep the repeat maths sane
#: (a zero interval would mean infinite rows per frame); it is deliberately far
#: below a frame, because the frame is not the speed limit -- the renderer decides
#: how many rows a frame may take. An earlier version floored this at 1/60, which
#: silently made every SDF above ~51 the same speed.
_MIN_SOFT_INTERVAL = 1e-4

#: Spellings players actually type, mapped to the names pygame knows.
_KEY_ALIASES = {
    'lctrl': 'left ctrl', 'rctrl': 'right ctrl', 'ctrl': 'left ctrl',
    'lshift': 'left shift', 'rshift': 'right shift', 'shift': 'left shift',
    'lalt': 'left alt', 'ralt': 'right alt', 'alt': 'left alt',
    'esc': 'escape', 'return': 'return', 'enter': 'return',
    'spacebar': 'space', ' ': 'space',
    'arrowleft': 'left', 'arrowright': 'right', 'arrowup': 'up',
    'arrowdown': 'down', 'uparrow': 'up', 'downarrow': 'down',
    'leftarrow': 'left', 'rightarrow': 'right',
    'del': 'delete', 'ins': 'insert', 'pgup': 'page up', 'pgdn': 'page down',
}


class ControlsError(ValueError):
    """A binding or timing value that cannot be used."""


def _ensure_pygame():
    """Initialise pygame's modules if the caller has not.

    ``pygame.key.key_code`` *works* before ``pygame.init()`` but warns that its
    answer may be wrong, and the settings are loaded before any window opens (the
    startup printout, the settings screen's first draw), so the warning would land
    on every run. ``init`` does not open a window -- only ``set_mode`` does -- so
    this is safe in the headless tests too.
    """
    if not pygame.get_init():
        try:
            pygame.init()
        except Exception:                     # noqa: BLE001 - never fatal here
            pass


def parse_key(name):
    """A key name from the config as a pygame key code.

    Accepts what ``pygame.key.name`` produces ('left ctrl', 'page up') plus the
    shorthands people actually write ('lctrl', 'esc', ' '). Raises
    :class:`ControlsError` with the offending name, because a silently ignored
    binding is worse than a startup error.
    """
    if isinstance(name, int):
        return name
    if not isinstance(name, str):
        raise ControlsError(f'not a key name: {name!r}')
    raw = name.strip().lower().replace('_', ' ')
    if not raw:
        # A bare space *is* the space bar; every other blank name is nothing. The
        # lookup has to happen here rather than after a blank-name rejection, or
        # the ' ' alias in the table above is unreachable.
        raw = name.lower()
    candidate = _KEY_ALIASES.get(raw, raw)
    if not candidate.strip():
        raise ControlsError(f'not a key name: {name!r}')
    _ensure_pygame()
    try:
        return pygame.key.key_code(candidate)
    except ValueError:
        raise ControlsError(f'unknown key name {name!r}') from None


def key_name(code):
    """The readable name of a key code, for the settings screen."""
    _ensure_pygame()
    try:
        return pygame.key.name(code)
    except Exception:
        return str(code)


def _as_seconds(ms, what):
    try:
        value = float(ms)
    except (TypeError, ValueError):
        raise ControlsError(f'{what} must be a number of milliseconds, '
                            f'got {ms!r}') from None
    if value < 0:
        raise ControlsError(f'{what} cannot be negative (got {ms!r})')
    return value / 1000.0


class InputSettings:
    """Bindings plus timing, in TETR.IO's units.

    The timing properties expose seconds because that is what the renderer's
    clocks use; everything the player types and the settings file stores stays in
    milliseconds and SDF multipliers.
    """

    def __init__(self, bindings=None, das_ms=None, arr_ms=None, sdf=None,
                 lock_delay_ms=None, lock_reset_limit=None,
                 soft_drop_instant=None, dcd_ms=None, warnings=()):
        merged = dict(DEFAULT_BINDINGS)
        for action, keys in (bindings or {}).items():
            if action not in DEFAULT_BINDINGS:
                continue                      # unknown action: ignored on load
            merged[action] = list(keys)
        self.bindings = merged
        timing = dict(DEFAULT_TIMING)
        for key, value in (('das_ms', das_ms), ('arr_ms', arr_ms), ('sdf', sdf),
                           ('lock_delay_ms', lock_delay_ms),
                           ('lock_reset_limit', lock_reset_limit),
                           ('soft_drop_instant', soft_drop_instant),
                           ('dcd_ms', dcd_ms)):
            if value is not None:
                timing[key] = value
        # Set before the clamps below, which append to it: assigning it afterwards
        # made every negative timing raise AttributeError instead of warning.
        self.warnings = list(warnings)
        self.das_ms = self._number(timing['das_ms'], 'das_ms')
        self.arr_ms = self._number(timing['arr_ms'], 'arr_ms')
        self.lock_delay_ms = self._number(timing['lock_delay_ms'], 'lock_delay_ms')
        self.lock_reset_limit = int(self._number(timing['lock_reset_limit'],
                                                 'lock_reset_limit'))
        self.sdf = (None if timing['sdf'] is None
                    else self._number(timing['sdf'], 'sdf'))
        self.soft_drop_instant = self._flag(timing['soft_drop_instant'],
                                            'soft_drop_instant')
        # No upper bound: DCD is a delay, and a player is free to pick a silly one.
        # Only the floor is enforced, and a junk value falls back in ``_number``.
        self.dcd_ms = self._number(timing['dcd_ms'], 'dcd_ms')
        # Negative timings are clamped here rather than raising later from a
        # property: a bad value in a config file should degrade to something
        # playable and say so, not crash on the first key press.
        for key in ('das_ms', 'arr_ms', 'lock_delay_ms', 'dcd_ms'):
            if getattr(self, key) < 0:
                self.warnings.append(f'{key} cannot be negative; using 0')
                setattr(self, key, 0.0)
        if self.sdf is not None and self.sdf < 0:
            self.warnings.append('sdf cannot be negative; using 0 (instant)')
            self.sdf = 0.0
        if self.lock_reset_limit < 0:
            self.warnings.append('lock_reset_limit cannot be negative; using 0')
            self.lock_reset_limit = 0
        self._codes = None

    # -- timing, converted once -------------------------------------------

    def _number(self, value, what):
        """A timing value as a float, or the documented default plus a warning.

        ``load`` and ``from_dict`` promise never to raise, and ``float('soon')``
        raises -- so a typo in a config file used to stop the game before it had
        opened a window, which is the opposite of what the clamping below is for.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            fallback = float(DEFAULT_TIMING[what])
            self.warnings.append(f'{what} is not a number ({value!r}); '
                                 f'using {fallback:g}')
            return fallback

    def _flag(self, value, what):
        """A yes/no setting, accepting the spellings a config file might use.

        ``bool('false')`` is True, so a string has to be read rather than coerced
        -- a config saying ``soft_drop_instant: "off"`` must not turn it on.
        """
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('true', 'yes', 'on', '1'):
                return True
            if text in ('false', 'no', 'off', '0', ''):
                return False
            self.warnings.append(f'{what} is not a yes/no value ({value!r}); '
                                 f'using {DEFAULT_TIMING[what]}')
            return bool(DEFAULT_TIMING[what])
        if isinstance(value, (bool, int, float)):
            return bool(value)
        self.warnings.append(f'{what} is not a yes/no value ({value!r}); '
                             f'using {DEFAULT_TIMING[what]}')
        return bool(DEFAULT_TIMING[what])

    @property
    def das(self):
        """Delay before horizontal auto-repeat, in seconds."""
        return _as_seconds(self.das_ms, 'das_ms')

    @property
    def arr(self):
        """Auto-repeat rate in seconds. 0 means instant."""
        return _as_seconds(self.arr_ms, 'arr_ms')

    @property
    def lock_delay(self):
        return _as_seconds(self.lock_delay_ms, 'lock_delay_ms')

    @property
    def dcd(self):
        """DAS Cut Delay, in seconds. 0 disables it.

        After a successful rotation, or after a new piece appears, horizontal
        auto-repeat stops firing for this long. The direction stays held and
        nothing is cancelled: the repeat simply waits, and DAS does not restart
        either -- the charge is frozen, not reset, so a key that was most of the
        way through its delay keeps that progress through the window.

        A *fresh press* is deliberately not affected. DCD cuts auto-repeat, not
        the first step, so a tap still moves one cell during the freeze; gating
        ``key_down`` instead would make it feel like dropped input rather than
        like precision.
        """
        return _as_seconds(self.dcd_ms, 'dcd_ms')

    @property
    def soft_instant(self):
        """True when soft drop teleports to the landing row instead of being fast.

        Three ways to ask for it, all meaning the same thing: the explicit
        ``soft_drop_instant`` flag (what the settings screen sets with ``I``, and
        what a player who wants it should use), ``sdf: 0``, or ``sdf: null`` in the
        config file. The last two predate the flag and keep working; note that the
        *constructor's* ``sdf=None`` means "unspecified" and falls back to the
        default, so the null-means-instant rule lives in :meth:`from_dict`, which is
        what reads a config.
        """
        return (self.soft_drop_instant or self.sdf is None or self.sdf <= 0)

    def soft_interval(self, level):
        """Seconds per soft-drop row at ``level``, or None when instant.

        SDF is a multiplier of the gravity interval, which is what TETR.IO means
        by it: a bigger factor is a faster soft drop, and 0 is unlimited.

        The value is returned *unclamped by the frame rate* on purpose. Flooring it
        at 1/60 made every SDF above about 51 identical -- 200x behaved exactly like
        60x -- because one row per frame was all the old caller could take. The
        caller now takes as many rows as the interval allows in one frame.
        """
        if self.soft_instant:
            return None
        from .render import gravity_interval
        return max(_MIN_SOFT_INTERVAL, gravity_interval(level) / self.sdf)

    # -- bindings ---------------------------------------------------------

    def codes(self, action):
        """The key codes bound to ``action``, in slot order."""
        return [parse_key(name) for name in self.bindings.get(action, ())]

    @property
    def key_map(self):
        """``{key code: action}`` for every binding (first binding wins a clash)."""
        if self._codes is None:
            table = {}
            for action in ACTION_NAMES:
                for code in self.codes(action):
                    table.setdefault(code, action)
            self._codes = table
        return self._codes

    def action_for(self, key):
        """The action a key code performs, or None."""
        return self.key_map.get(key)

    def binding_names(self, action):
        return [key_name(code) for code in self.codes(action)]

    def set_binding(self, action, slot, name):
        """Put ``name`` in ``slot`` (0 or 1) for ``action``, clearing a clash.

        A key bound to something else is removed from there rather than silently
        doing two things at once, which is what a player expects when they rebind
        onto an existing key.
        """
        if action not in self.bindings:
            raise ControlsError(f'unknown action {action!r}')
        code = parse_key(name)
        for other in ACTION_NAMES:
            if other == action:
                continue
            kept = [c for c in self.codes(other) if c != code]
            self.bindings[other] = [key_name(c) for c in kept]
        # Drop the key from this action's *other* slot too, or binding slot 1 to a
        # key already in slot 0 leaves the same key twice and ``problems()``
        # reports it as a clash with itself.
        keys = [k for k in self.bindings.get(action, [])
                if parse_key(k) != code]
        while len(keys) <= slot:
            keys.append(None)
        keys[slot] = key_name(code)
        self.bindings[action] = [k for k in keys if k]
        self._codes = None
        return self

    def clear_binding(self, action, slot):
        keys = list(self.bindings.get(action, []))
        if 0 <= slot < len(keys):
            del keys[slot]
        self.bindings[action] = keys
        self._codes = None
        return self

    def problems(self):
        """Human-readable warnings: unparsable names and keys doing two jobs."""
        out = []
        seen = {}
        for action in ACTION_NAMES:
            names = self.bindings.get(action, [])
            if len(names) > MAX_BINDINGS:
                out.append(f'{action}: {len(names)} bindings, only '
                           f'{MAX_BINDINGS} are used')
            for name in names:
                try:
                    code = parse_key(name)
                except ControlsError as exc:
                    out.append(f'{action}: {exc}')
                    continue
                if code in seen:
                    out.append(f'{key_name(code)} is bound to both {seen[code]} '
                               f'and {action}')
                else:
                    seen[code] = action
        return out

    # -- serialisation ----------------------------------------------------

    def to_dict(self):
        return {
            'bindings': {a: list(self.bindings.get(a, [])) for a in ACTION_NAMES},
            'das_ms': self.das_ms,
            'arr_ms': self.arr_ms,
            'sdf': self.sdf,
            'soft_drop_instant': self.soft_drop_instant,
            'lock_delay_ms': self.lock_delay_ms,
            'lock_reset_limit': self.lock_reset_limit,
            'dcd_ms': self.dcd_ms,
        }

    @classmethod
    def from_dict(cls, data):
        """Build from a config-style mapping, collecting warnings as it goes."""
        if not isinstance(data, dict):
            return cls(warnings=['input: section is not a mapping; using defaults'])
        warnings = []
        raw = data.get('bindings') or {}
        bindings = {}
        for action, keys in raw.items():
            if action not in DEFAULT_BINDINGS:
                warnings.append(f'input.bindings: unknown action {action!r}')
                continue
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, (list, tuple)):
                warnings.append(f'input.bindings.{action}: expected a key or a '
                                f'list of keys, got {keys!r}')
                continue
            good = []
            for name in keys[:MAX_BINDINGS]:
                try:
                    parse_key(name)
                except ControlsError as exc:
                    warnings.append(f'input.bindings.{action}: {exc}')
                    continue
                good.append(str(name).strip().lower())
            bindings[action] = good
        # ``sdf:`` present but empty means instant; absent means the default. The
        # two used to be the same thing because the constructor reads None as
        # "unspecified", which quietly made the documented ``sdf: null`` do
        # nothing at all.
        sdf = data.get('sdf')
        if 'sdf' in data and sdf is None:
            sdf = 0.0
        settings = cls(bindings=bindings,
                       das_ms=data.get('das_ms'), arr_ms=data.get('arr_ms'),
                       sdf=sdf,
                       soft_drop_instant=data.get('soft_drop_instant'),
                       lock_delay_ms=data.get('lock_delay_ms'),
                       lock_reset_limit=data.get('lock_reset_limit'),
                       dcd_ms=data.get('dcd_ms'),
                       warnings=warnings)
        return settings


def settings_path(directory=None):
    """Where the settings screen writes, next to ``config.yaml``."""
    if directory is None:
        from . import PROJECT_DIR
        directory = PROJECT_DIR
    return os.path.join(directory, SETTINGS_NAME)


def load(config=None, directory=None):
    """Build the settings in force: defaults, then config, then settings.yaml.

    Never raises on bad input. A broken key name becomes a warning and that one
    binding falls back to its default, because refusing to start the game over a
    typo in a key name would be a worse failure than the typo.
    """
    if config is None:
        from . import load_config
        config = load_config()
    settings = InputSettings.from_dict(config.get('input') or {})
    path = settings_path(directory)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                saved = yaml.safe_load(handle) or {}
        except Exception as exc:
            settings.warnings.append(f'could not read {SETTINGS_NAME} ({exc})')
            return settings
        merged = settings.to_dict()
        block = saved.get('input', saved) if isinstance(saved, dict) else {}
        for key, value in (block or {}).items():
            if key == 'bindings' and isinstance(value, dict):
                for action, keys in value.items():
                    if action in DEFAULT_BINDINGS:
                        merged['bindings'][action] = keys
            elif key in merged:
                merged[key] = value
        settings = InputSettings.from_dict(merged)
    return settings


def save(settings, directory=None):
    """Write the settings to ``settings.yaml``. Returns the path written."""
    path = settings_path(directory)
    payload = {'input': settings.to_dict()}
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# Player input settings, written by the in-game settings '
                     'screen.\n'
                     '# Overrides the input: section of config.yaml. '
                     'DAS/ARR/lock delay in ms;\n'
                     '# sdf is a gravity multiplier (0 = instant). Two keys per '
                     'action.\n')
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path
