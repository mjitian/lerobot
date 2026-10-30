import os
import time
import platform
import logging
from typing import Dict, Any, Optional


def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
        if not blocking:
            cmd += " &"
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
        if blocking:
            cmd += "  --wait"
    elif platform.system() == "Windows":
        # TODO(rcadene): Make blocking option work for Windows
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    os.system(cmd)


def log_say(text, play_sounds, blocking=False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


_frequency_trackers: Dict[str, Dict[str, Any]] = {}

def track_frequency(tracker_id: str, 
                    print_interval: float = 5.0, 
                    message_template: str = "{tracker_id} frequency: {rate:.2f} Hz ({count} calls in {elapsed:.2f} seconds)",
                    log_level: str = "debug") -> Optional[float]:

    global _frequency_trackers
    if tracker_id not in _frequency_trackers:
        _frequency_trackers[tracker_id] = {
            "counter": 0,
            "last_time": time.time(),
            "last_print": time.time(),
            "print_interval": print_interval
        }

    tracker = _frequency_trackers[tracker_id]
    tracker["counter"] += 1
    current_time = time.time()

    if current_time - tracker["last_print"] >= tracker["print_interval"]:
        elapsed = current_time - tracker["last_time"]
        current_rate = None
        if elapsed > 0:
            current_rate = tracker["counter"] / elapsed
            message = message_template.format(
                tracker_id=tracker_id,
                rate=current_rate,
                count=tracker["counter"],
                elapsed=elapsed
            )
            if log_level == "debug":
                logging.debug(message)
            elif log_level == "info":
                logging.info(message)
            elif log_level == "warning": 
                logging.warning(message)
            elif log_level == "error":
                logging.error(message)
            else:
                logging.info(message)
            print(message)
        tracker["counter"] = 0
        tracker["last_time"] = current_time
        tracker["last_print"] = current_time

        return current_rate

    return None