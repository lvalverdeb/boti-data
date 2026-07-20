"""Safely injects dynamically-built ORM classes into sys.modules.

Split out of SqlModelRegistry: this is self-contained module-injection
mechanics with no dependency on the registry's caching/locking state beyond
a single lock, so it moves here wholesale as a free function.
"""

from __future__ import annotations

import sys
import threading
import types


def register_as_module_attribute(
    cls: type,
    module_name: str,
    class_name: str,
    *,
    lock: threading.Lock,
    sentinel: str,
) -> None:
    """Injects the class into sys.modules maintaining pickling safety across processes."""
    parts = module_name.split(".")
    with lock:
        existing = sys.modules.get(module_name)
        if existing is not None and not getattr(existing, sentinel, False):
            raise ValueError(
                "module_label must target an unused or registry-managed dynamic module namespace."
            )

        for index in range(1, len(parts) + 1):
            current_name = ".".join(parts[:index])
            current_module = sys.modules.get(current_name)
            if current_module is None:
                current_module = types.ModuleType(current_name)
                current_module.__package__ = current_name.rpartition(".")[0]
                current_module.__path__ = []  # type: ignore[attr-defined]
                setattr(current_module, sentinel, True)
                sys.modules[current_name] = current_module
            elif index == len(parts) and not getattr(current_module, sentinel, False):
                raise ValueError(
                    "module_label must target an unused or registry-managed dynamic module namespace."
                )

            if index > 1:
                parent_name = ".".join(parts[: index - 1])
                child_name = parts[index - 1]
                setattr(sys.modules[parent_name], child_name, current_module)

        setattr(sys.modules[module_name], class_name, cls)
