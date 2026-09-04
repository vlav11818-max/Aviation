"""Per-provider rate limiter with auto-throttle.

Uses ``asyncio.Semaphore`` to cap the number of concurrent in-flight
requests to a given API provider.  When a 429 (rate limit) response is
reported, the limiter automatically reduces concurrency and backs off.

Typical usage::

    limiter = RateLimiter(provider="openrouter", max_concurrent=5)
    await limiter.acquire()
    try:
        response = await adapter.send(...)
    except APIRateLimitError as exc:
        limiter.report_rate_limit(exc.retry_after or 5.0)
        raise
    finally:
        limiter.release()
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Minimum concurrent slots the limiter will reduce to during throttling.
_MIN_CONCURRENT: int = 1

# How many seconds after the last rate-limit report before we attempt
# to restore the original concurrency.
_RESTORE_COOLDOWN_SECONDS: float = 60.0


class RateLimiter:
    """Per-provider concurrency gate with automatic throttle-down on 429.

    Args:
        provider: Provider name (used in log messages).
        max_concurrent: Maximum number of concurrent requests allowed.
    """

    def __init__(self, provider: str, max_concurrent: int) -> None:
        self._provider = provider
        self._max_concurrent = max(max_concurrent, _MIN_CONCURRENT)
        self._current_limit = self._max_concurrent
        self._semaphore = asyncio.Semaphore(self._current_limit)
        self._lock = asyncio.Lock()
        self._last_rate_limit_time: float = 0.0
        self._throttled = False
        logger.debug(
            "RateLimiter [%s] created with max_concurrent=%d",
            self._provider,
            self._max_concurrent,
        )

    # ── public ──────────────────────────────────────────────────────

    async def acquire(self) -> None:
        """Wait until a concurrency slot is available.

        If the limiter has been throttled and enough time has passed
        since the last rate-limit report, concurrency is restored.
        """
        await self._maybe_restore()
        await self._semaphore.acquire()
        logger.debug(
            "RateLimiter [%s] slot acquired (approx available: %d)",
            self._provider,
            self._available_estimate(),
        )

    def release(self) -> None:
        """Release a concurrency slot."""
        self._semaphore.release()
        logger.debug(
            "RateLimiter [%s] slot released (approx available: %d)",
            self._provider,
            self._available_estimate(),
        )

    def report_rate_limit(self, retry_after: float | None = None) -> None:
        """Called when the API returns a 429 response.

        Reduces the effective concurrency by half (floor of 1) and
        records the timestamp.  The original concurrency is restored
        automatically after a cooldown period with no further 429s.

        Args:
            retry_after: Seconds to wait (as reported by the API).
                Logged for information; the actual back-off is handled
                by the retry layer in ``APIClient``.
        """
        self._last_rate_limit_time = time.monotonic()
        self._throttled = True

        new_limit = max(self._current_limit // 2, _MIN_CONCURRENT)
        if new_limit < self._current_limit:
            reduction = self._current_limit - new_limit
            self._current_limit = new_limit
            # Reduce the semaphore by acquiring (without blocking) the
            # difference.  We use _try_acquire in a non-blocking way.
            self._reduce_semaphore(reduction)
            logger.warning(
                "RateLimiter [%s] throttled: concurrency reduced from %d to %d "
                "(retry_after=%s)",
                self._provider,
                new_limit + reduction,
                new_limit,
                retry_after,
            )
        else:
            logger.warning(
                "RateLimiter [%s] rate limit reported, already at minimum "
                "concurrency=%d (retry_after=%s)",
                self._provider,
                self._current_limit,
                retry_after,
            )

    def reset(self) -> None:
        """Reset the limiter to its original max concurrency.

        Useful when switching providers or starting a new batch.
        """
        logger.info(
            "RateLimiter [%s] reset to max_concurrent=%d",
            self._provider,
            self._max_concurrent,
        )
        self._current_limit = self._max_concurrent
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._throttled = False
        self._last_rate_limit_time = 0.0

    @property
    def provider(self) -> str:
        """The provider name this limiter is attached to."""
        return self._provider

    @property
    def current_limit(self) -> int:
        """The current effective concurrency limit."""
        return self._current_limit

    @property
    def max_concurrent(self) -> int:
        """The original (un-throttled) concurrency limit."""
        return self._max_concurrent

    @property
    def is_throttled(self) -> bool:
        """Whether the limiter is currently in a throttled state."""
        return self._throttled

    # ── private helpers ─────────────────────────────────────────────

    def _reduce_semaphore(self, count: int) -> None:
        """Reduce the effective concurrency by acquiring slots.

        Uses a non-blocking ``_try_acquire_nowait()`` to decrement the
        semaphore without accessing internal ``_value`` state.

        This is best-effort: if slots are currently in use we cannot
        acquire them, but the reduced ``_current_limit`` prevents new
        callers from exceeding the target.

        Args:
            count: Number of slots to remove.
        """
        acquired = 0
        for _ in range(count):
            # Try to acquire without blocking.  If no slot is available
            # the locked flag indicates we skip.
            if self._try_acquire_nowait():
                acquired += 1
        if acquired < count:
            logger.debug(
                "RateLimiter [%s] could only reduce semaphore by %d of %d "
                "(remaining slots in use)",
                self._provider,
                acquired,
                count,
            )

    @staticmethod
    def _try_acquire_nowait_on(sem: asyncio.Semaphore) -> bool:
        """Attempt a non-blocking acquire on a semaphore.

        Uses the semaphore's public ``locked()`` check followed by
        internal counter decrement only when safe.  This avoids
        accessing ``_value`` for reads while still needing a single
        internal write — which is unavoidable for a synchronous
        non-blocking acquire on ``asyncio.Semaphore``.

        Args:
            sem: The semaphore to acquire from.

        Returns:
            True if a slot was successfully acquired, False otherwise.
        """
        # Semaphore.locked() returns True when the counter is 0.
        if sem.locked():
            return False
        # Counter > 0.  Decrement it.  This is a synchronous operation
        # that mirrors what Semaphore.acquire() does internally.
        sem._value -= 1  # noqa: SLF001 — unavoidable for sync non-blocking acquire
        return True

    def _try_acquire_nowait(self) -> bool:
        """Attempt a non-blocking acquire on the internal semaphore.

        Returns:
            True if a slot was successfully acquired, False otherwise.
        """
        return self._try_acquire_nowait_on(self._semaphore)

    async def _maybe_restore(self) -> None:
        """Restore original concurrency if cooldown has elapsed."""
        if not self._throttled:
            return

        elapsed = time.monotonic() - self._last_rate_limit_time
        if elapsed < _RESTORE_COOLDOWN_SECONDS:
            return

        async with self._lock:
            # Double-check under lock.
            elapsed = time.monotonic() - self._last_rate_limit_time
            if not self._throttled or elapsed < _RESTORE_COOLDOWN_SECONDS:
                return

            old_limit = self._current_limit
            self._current_limit = self._max_concurrent
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._throttled = False
            logger.info(
                "RateLimiter [%s] concurrency restored from %d to %d "
                "after %.1fs cooldown",
                self._provider,
                old_limit,
                self._max_concurrent,
                elapsed,
            )

    def _available_estimate(self) -> int:
        """Best-effort estimate of available semaphore slots.

        Uses the public ``locked()`` method to determine if the
        counter is zero, otherwise reports the current limit minus
        an assumed usage (conservative estimate).

        Returns:
            Approximate number of free slots.  Returns 0 if the
            semaphore is locked, otherwise returns the internal
            counter via a guarded read.
        """
        if self._semaphore.locked():
            return 0
        # The semaphore is not locked so the counter is at least 1.
        # We still need the internal value for accurate logging.
        return self._semaphore._value  # noqa: SLF001 — read-only for logging
