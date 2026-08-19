# fuzz_ui.py — shutdown / pause / log level

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from enum import IntEnum

import api_data_store as ds
from api_common import set_logger


class ShutdownLevel(IntEnum):
    NONE = 0
    BUDGET = 1  # Stop after current budget block (coverage band)
    STRATEGY = 2  # Stop after current strategy
    SAMPLE = 3  # Stop after current test
    IMMEDIATE = 4  # Exit immediately


LOG_LEVELS = ["SUCCESS", "INFO", "DEBUG", "TRACE"]
_current_log_level_index = 1

_shutdown_level = ShutdownLevel.NONE
_shutdown_state_index = 0

_pause_after_test = False
_paused = False
_pause_condition = threading.Condition()

_ui_root: tk.Tk | None = None

SHUTDOWN_STATES = [
    (ShutdownLevel.BUDGET, "Graceful: Budget", "orange"),
    (ShutdownLevel.STRATEGY, "Graceful: Strategy", "darkorange"),
    (ShutdownLevel.SAMPLE, "Graceful: Test", "orangered"),
    (ShutdownLevel.IMMEDIATE, "Immediate: Exit", "red"),
]


def get_shutdown_level() -> ShutdownLevel:
    return _shutdown_level


def _create_ui() -> None:
    global _ui_root

    root = tk.Tk()
    root.title("Test Control")
    root.geometry("250x180")
    root.attributes("-topmost", True)

    # --- 1. SHUTDOWN BUTTON LOGIC ---
    def on_shutdown_click():
        global _shutdown_level, _shutdown_state_index

        _shutdown_state_index += 1

        if _shutdown_state_index > len(SHUTDOWN_STATES):
            print("\nHard exit — terminating immediately.", file=sys.stderr)
            os._exit(1)

        level, _text, _bg = SHUTDOWN_STATES[_shutdown_state_index - 1]
        _shutdown_level = level
        print(f"\nShutdown requested — finishing current {level.name.lower()}...", file=sys.stderr)

        if _shutdown_state_index < len(SHUTDOWN_STATES):
            _, next_text, next_bg = SHUTDOWN_STATES[_shutdown_state_index]
            shutdown_btn.config(text=next_text, bg=next_bg)
        else:
            shutdown_btn.config(text="FORCE TERMINATE", bg="red")

    _, initial_text, initial_bg = SHUTDOWN_STATES[0]
    shutdown_btn = tk.Button(root, text=initial_text, command=on_shutdown_click, bg=initial_bg)
    shutdown_btn.pack(expand=True, fill="both", padx=10, pady=(10, 5))

    # --- 2. PAUSE AFTER TEST BUTTON LOGIC ---
    def on_pause_click():
        global _pause_after_test, _paused

        with _pause_condition:
            if _paused:
                _paused = False
                _pause_after_test = False
                pause_btn.config(text="Pause After Test", bg="lightblue")
                print("\n[CONTROL] Resumed — continuing tests...", file=sys.stderr)
                _pause_condition.notify_all()
            else:
                _pause_after_test = True
                pause_btn.config(text="Pause Requested...", bg="gold")
                print(
                    "\n[CONTROL] Pause after current test requested — will pause when test finishes...",
                    file=sys.stderr,
                )

    pause_btn = tk.Button(root, text="Pause After Test", command=on_pause_click, bg="lightblue")
    pause_btn.pack(expand=True, fill="both", padx=10, pady=5)
    root.pause_btn = pause_btn  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]

    # --- 3. LOG LEVEL TOGGLE LOGIC ---
    def on_log_click():
        global _current_log_level_index
        _current_log_level_index = (_current_log_level_index + 1) % len(LOG_LEVELS)
        new_level = LOG_LEVELS[_current_log_level_index]

        log_btn.config(text=f"Log Level: {new_level}")

        try:
            cfg = ds.get_config("config")
            log_format = cfg.get("logger", {}).get("format", "<level>{message}</level>")
        except Exception:  # noqa: BLE001
            log_format = "<level>{message}</level>"

        set_logger({"logger": {"level": new_level, "format": log_format}})
        print(f"[CONTROL] Loguru runtime level updated to: {new_level}", file=sys.stdout)

    initial_log_text = f"Log Level: {LOG_LEVELS[_current_log_level_index]}"
    log_btn = tk.Button(root, text=initial_log_text, command=on_log_click, bg="lightgray")
    log_btn.pack(expand=True, fill="both", padx=10, pady=(5, 10))

    _ui_root = root
    root.mainloop()


def _set_paused_ui_state() -> None:
    """Update the pause button to the 'Paused / click to Resume' state (Tk thread only)."""
    if _ui_root is None:
        return
    try:
        btn = getattr(_ui_root, "pause_btn", None)
        if btn is not None:
            btn.config(text="▶ RESUME", bg="limegreen")
    except Exception:  # noqa: BLE001  # ruff: ignore[try-except-pass]
        pass


def wait_if_pause_requested() -> None:
    """
    If a pause-after-test was requested, block until Resume (or Immediate shutdown).
    Safe to call from the main fuzzer thread.
    """
    global _pause_after_test, _paused

    with _pause_condition:
        if not _pause_after_test:
            return

        _paused = True
        _pause_after_test = False
        print("\n[CONTROL] Paused after test — click RESUME to continue...", file=sys.stderr)

        if _ui_root is not None:
            try:
                _ui_root.after(0, _set_paused_ui_state)
            except Exception:  # noqa: BLE001  # ruff: ignore[try-except-pass]
                pass

        while _paused:
            _pause_condition.wait(timeout=0.5)
            if _shutdown_level >= ShutdownLevel.IMMEDIATE:
                _paused = False
                break


def install_control_ui() -> None:
    """Read startup log level from config and launch the control UI on a daemon thread."""
    global _current_log_level_index

    try:
        startup_config = ds.get_config("config")
        startup_level = startup_config.get("logger", {}).get("level", "INFO").upper()
        if startup_level in LOG_LEVELS:
            _current_log_level_index = LOG_LEVELS.index(startup_level)
    except Exception as e:  # noqa: BLE001
        print(
            f"[CONTROL WARNING] Failed to read startup config level: {e}. Defaulting index to INFO.",
            file=sys.stderr,
        )

    t = threading.Thread(target=_create_ui, daemon=True)
    t.start()
