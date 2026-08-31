"""Fire-and-forget helper for background coroutines (e.g. shadow cache
verification) that must not add latency to the response being served.

asyncio.create_task() alone isn't enough: if nothing keeps a reference to
the Task, it can be garbage-collected mid-execution (this is a documented
asyncio gotcha, not a hypothetical one). Keeping tasks in a module-level
set until they finish is the standard fix.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any

_background_tasks: set[asyncio.Task] = set()


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
