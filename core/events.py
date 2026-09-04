"""Thread-safe event system for core↔GUI communication.

The EventBus decouples core logic from the GUI. Core workers emit events
(step progress, errors, cost updates). The GUI polls events in its main
loop via ``after()``.

Typical usage::

    bus = EventBus()
    bus.subscribe(EventType.STEP_COMPLETED, my_callback)
    bus.emit(EventType.STEP_COMPLETED, step="concept", topic="Ancient Temple")

    # In the GUI after() loop — drain ALL pending events each tick:
    events = bus.poll_all()
    for event in events:
        handle(event)
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Maximum number of events drained per poll_all() call.
# This prevents the GUI thread from freezing when a large burst of
# events has accumulated (e.g. 5 parallel workers all completing
# sections simultaneously).  Any events beyond this cap remain in the
# queue and are processed on the next tick (100 ms later).
# At 100 ms polling and ~50 events/tick max, the queue drains at
# ~500 events/second — far above any realistic production throughput.
_MAX_EVENTS_PER_TICK: int = 50

# Queue depth at which a warning is logged.  Sustained depth above this
# level indicates the GUI is falling behind producers and the poll
# interval or cap may need tuning.
_QUEUE_DEPTH_WARN_THRESHOLD: int = 100


class EventType(Enum):
    """All event types emitted by the system."""

    # Progress
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    SECTION_COMPLETED = "section_completed"
    STORY_COMPLETED = "story_completed"
    BATCH_COMPLETED = "batch_completed"

    # Status
    EVALUATION_RESULT = "evaluation_result"
    REVISION_STARTED = "revision_started"

    # Errors
    API_ERROR = "api_error"
    API_FALLBACK = "api_fallback"
    STEP_FAILED = "step_failed"

    # Cost
    COST_UPDATE = "cost_update"

    # Log
    LOG_MESSAGE = "log_message"


@dataclass(frozen=True)
class Event:
    """Immutable event carrying data from core to GUI.

    Attributes:
        type: The category of this event.
        data: Arbitrary key-value payload.
        timestamp: Unix timestamp when the event was created.
    """

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe event bus.

    Core pushes events via ``emit()``, GUI reads them via ``poll_all()``.
    Optional ``subscribe()`` registers callbacks that fire on ``emit()``.

    The internal queue is unbounded so that fast producers never block.
    Consumers must call ``poll_all()`` — not ``poll()`` — in their tick
    loop so that all pending events are processed each tick, preventing
    UI lag when multiple workers fire events simultaneously.

    Per-tick event processing is capped at ``_MAX_EVENTS_PER_TICK``
    (default 50) to keep the GUI thread responsive even when a large
    burst of events accumulates.  Events beyond the cap are carried
    forward to the next tick automatically.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}
        logger.debug("EventBus initialised")

    def emit(self, event_type: EventType, **data: Any) -> None:
        """Create an event and push it onto the queue.

        Also invokes any registered callbacks for the event type
        synchronously on the calling thread.

        Args:
            event_type: The type of event to emit.
            **data: Arbitrary key-value pairs included in the event payload.
        """
        event = Event(type=event_type, data=data)
        self._queue.put(event)

        # Warn if the queue is growing unusually large.
        depth = self._queue.qsize()
        if depth >= _QUEUE_DEPTH_WARN_THRESHOLD:
            logger.warning(
                "EventBus queue depth is %d — GUI may be falling behind "
                "producers (threshold=%d). Consider reducing parallel "
                "workers or increasing poll frequency.",
                depth,
                _QUEUE_DEPTH_WARN_THRESHOLD,
            )

        callbacks = self._subscribers.get(event_type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "Subscriber callback failed for event %s",
                    event_type.value,
                )

        logger.debug("Event emitted: %s (queue depth: %d)", event_type.value, depth)

    def poll(self) -> Event | None:
        """Non-blocking read of the next single event.

        Prefer ``poll_all()`` in GUI tick loops to avoid processing
        only one event per 100 ms tick when many are pending.

        Returns:
            The next Event in the queue, or ``None`` if the queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def poll_all(self, max_events: int = _MAX_EVENTS_PER_TICK) -> list[Event]:
        """Drain pending events from the queue up to ``max_events``.

        This is the correct method to call from a GUI ``after()`` tick
        loop.  It processes all (or up to ``max_events``) queued events
        in a single call, ensuring the GUI stays current even when
        multiple workers fire events simultaneously.

        The cap prevents the GUI thread from freezing when a large burst
        accumulates: any events beyond ``max_events`` remain in the queue
        and are processed on the next tick.

        Args:
            max_events: Maximum number of events to drain in this call.
                Defaults to ``_MAX_EVENTS_PER_TICK`` (50).  Pass
                ``-1`` to drain all events with no cap (use only when
                the caller can guarantee the queue depth is bounded).

        Returns:
            A list of drained events (may be empty).
        """
        events: list[Event] = []
        limit = max_events if max_events > 0 else None

        while True:
            if limit is not None and len(events) >= limit:
                remaining = self._queue.qsize()
                if remaining > 0:
                    logger.debug(
                        "poll_all: cap reached (%d events drained), "
                        "%d events deferred to next tick",
                        len(events),
                        remaining,
                    )
                break

            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if events:
            logger.debug("poll_all: drained %d events", len(events))

        return events

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """Register a callback invoked each time *event_type* is emitted.

        Callbacks are invoked **synchronously** inside ``emit()`` on the
        emitting thread, so they must be fast and thread-safe.

        Args:
            event_type: The event type to listen for.
            callback: A callable accepting a single ``Event`` argument.
        """
        self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug(
            "Subscriber registered for %s: %s",
            event_type.value,
            callback.__name__ if hasattr(callback, "__name__") else repr(callback),
        )

    def queue_depth(self) -> int:
        """Return the current number of unprocessed events in the queue.

        Useful for diagnostics and tests.

        Returns:
            Approximate queue depth (may be slightly stale in concurrent
            contexts due to ``queue.Queue.qsize()`` semantics).
        """
        return self._queue.qsize()

    def clear(self) -> None:
        """Discard all pending events in the queue.

        Useful when aborting a generation batch — stale events from the
        previous run should not be processed by a new run's handlers.
        """
        discarded = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        logger.debug("EventBus queue cleared (%d events discarded)", discarded)
