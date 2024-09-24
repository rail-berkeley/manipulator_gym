# abc abstract base class
from pynput import keyboard
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Any
import numpy as np


class ControlModule(ABC):
    """
    Abstract class for control module.

    This class provides an interface for control module.
    e.g. Keyboard, SpaceMouse, Oculus controller, etc.
    """

    @abstractmethod
    def get_action(self) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError

    @abstractmethod
    def register_callback(self, callback: Callable[[Any], None]):
        raise NotImplementedError


class KeyboardInputControl(ControlModule):
    def __init__(self, translation_diff=0.01, rotation_diff=0.01):
        keyboard_listener = keyboard.Listener(
            on_press=self._on_press_fn, on_release=self._on_release_fn
        )
        keyboard_listener.start()
        self._curr_action = np.zeros(6)
        self.key_map = {
            "w": (0, translation_diff),
            "s": (0, -translation_diff),
            "a": (1, translation_diff),
            "d": (1, -translation_diff),
            "z": (2, translation_diff),
            "c": (2, -translation_diff),
            "i": (3, rotation_diff),
            "k": (3, -rotation_diff),
            "j": (4, rotation_diff),
            "l": (4, -rotation_diff),
            "n": (5, rotation_diff),
            "m": (5, -rotation_diff),
        }
        self.pressed_keys = set()
        self._callback_fn = None
        self._changed = False

    def get_action(self) -> Tuple[np.ndarray, Any]:
        return self._curr_action, self.pressed_keys

    def register_callback(self, callback: Callable[[Any], None]):
        self._callback_fn = callback

    def _on_press_fn(self, key):
        # other wasd and zc for up and down
        if hasattr(key, "char") and key.char in self.key_map:
            idx, val = self.key_map[key.char]
            self._curr_action[idx] = val

        self.pressed_keys.add(key)
        if self._callback_fn is not None:
            self._callback_fn(key)

    def _on_release_fn(self, key):
        if hasattr(key, "char") and key.char in self.key_map:
            idx, val = self.key_map[key.char]
            self._curr_action[idx] = 0.0

        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        if self._callback_fn is not None:
            self._callback_fn(key)


if __name__ == "__main__":
    import time

    keyboard_input = KeyboardInputControl()

    while True:
        start = time.time()
        action, pressed = keyboard_input.get_action()
        print("used time: ", time.time() - start)
        print(action, pressed)
        time.sleep(0.1)
