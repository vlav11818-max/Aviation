"""Integration tests for ``core.recovery_manager``.

Tests cover: save batch state, load batch state, simulate crash
mid-batch (save state → new RecoveryManager → verify recovery
options), clear recovery state, and recovery with completed +
in_progress + failed + queued topics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import StateError
from core.recovery_manager import (
    BatchState,
    ConfigSnapshot,
    FailedTopic,
    RecoveryAction,
    RecoveryManager,
    RecoveryOptions,
    TopicProgress,
)
from core.settings import Settings


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def recovery_settings(tmp_dir: Path) -> Settings:
    """Build a ``Settings`` instance pointing recovery dir to tmp_dir."""
    content = {
        "paths": {
            "output_dir": str(tmp_dir / "output"),
            "data_dir": str(tmp_dir / "data"),
            "resources_dir": "resources",
            "recovery_dir": str(tmp_dir / "data" / "recovery"),
            "cache_dir": str(tmp_dir / "data" / "cache"),
            "analytics_dir": str(tmp_dir / "data" / "analytics"),
        },
        "cache": {"enabled": True, "skip_processed": True},
    }
    return Settings.model_validate(content)


@pytest.fixture()
def rm(recovery_settings: Settings) -> RecoveryManager:
    """Return a RecoveryManager pointing to a temp directory."""
    return RecoveryManager(recovery_settings)


@pytest.fixture()
def sample_batch() -> BatchState:
    """Return a BatchState with 5 topics all queued."""
    return RecoveryManager.create_batch_state(
        topics=["Topic A", "Topic B", "Topic C", "Topic D", "Topic E"],
        language="en",
        target_words=3000,
        tone="dramatic_cinematic",
        model="gpt-4o",
        provider="openrouter",
    )


# ── Tests: save / load round-trip ─────────────────────────────────────


class TestSaveLoad:
    """Tests for basic save and load operations."""

    def test_save_and_load(self, rm: RecoveryManager, sample_batch: BatchState) -> None:
        """Saved batch state should be loadable."""
        rm.save_batch_state(sample_batch)
        loaded = rm.load_batch_state()

        assert loaded.batch_id == sample_batch.batch_id
        assert loaded.total_topics == 5
        assert len(loaded.queued) == 5
        assert loaded.queued == sample_batch.queued

    def test_save_updates_timestamp(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """save_batch_state should update the updated_at timestamp."""
        original_updated = sample_batch.updated_at
        rm.save_batch_state(sample_batch)
        assert sample_batch.updated_at >= original_updated

    def test_load_nonexistent_raises(self, rm: RecoveryManager) -> None:
        """Loading without a saved state should raise StateError."""
        with pytest.raises(StateError, match="not found"):
            rm.load_batch_state()

    def test_config_snapshot_preserved(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """Config snapshot should survive round-trip."""
        rm.save_batch_state(sample_batch)
        loaded = rm.load_batch_state()

        assert loaded.config_snapshot.language == "en"
        assert loaded.config_snapshot.target_words == 3000
        assert loaded.config_snapshot.tone == "dramatic_cinematic"
        assert loaded.config_snapshot.model == "gpt-4o"
        assert loaded.config_snapshot.provider == "openrouter"


# ── Tests: has_unfinished ─────────────────────────────────────────────


class TestHasUnfinished:
    """Tests for the has_unfinished check."""

    def test_no_state_file_returns_false(self, rm: RecoveryManager) -> None:
        """No recovery file should return False."""
        assert rm.has_unfinished() is False

    def test_with_queued_topics(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """Queued topics should make has_unfinished return True."""
        rm.save_batch_state(sample_batch)
        assert rm.has_unfinished() is True

    def test_all_completed_returns_false_unless_failed(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """All completed + no queued + no in_progress + no failed = False."""
        sample_batch.completed = list(sample_batch.queued)
        sample_batch.queued = []
        sample_batch.in_progress = {}
        sample_batch.failed = []
        rm.save_batch_state(sample_batch)

        assert rm.has_unfinished() is False

    def test_with_failed_topics_returns_true(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """Failed topics should make has_unfinished return True."""
        sample_batch.completed = ["Topic A", "Topic B", "Topic C", "Topic D"]
        sample_batch.queued = []
        sample_batch.in_progress = {}
        sample_batch.failed = [
            FailedTopic(topic="Topic E", error="API timeout", stage="concept")
        ]
        rm.save_batch_state(sample_batch)

        assert rm.has_unfinished() is True


# ── Tests: get_recovery_options ───────────────────────────────────────


class TestRecoveryOptions:
    """Tests for recovery option generation."""

    def test_options_with_queued_and_failed(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """Options should include CONTINUE, RESTART, and RETRY_ERRORS."""
        sample_batch.completed = ["Topic A"]
        sample_batch.queued = ["Topic C", "Topic D", "Topic E"]
        sample_batch.in_progress = {}
        sample_batch.failed = [
            FailedTopic(topic="Topic B", error="API error", stage="outline")
        ]
        rm.save_batch_state(sample_batch)

        opts = rm.get_recovery_options()

        assert isinstance(opts, RecoveryOptions)
        assert opts.batch_id == sample_batch.batch_id
        assert opts.total_topics == 5
        assert opts.completed_count == 1
        assert opts.failed_count == 1
        assert opts.queued_count == 3
        assert RecoveryAction.CONTINUE in opts.available_actions
        assert RecoveryAction.RESTART in opts.available_actions
        assert RecoveryAction.RETRY_ERRORS in opts.available_actions

    def test_options_only_failed(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """With only failed topics, CONTINUE should NOT be available."""
        sample_batch.completed = ["Topic A", "Topic B", "Topic C", "Topic D"]
        sample_batch.queued = []
        sample_batch.in_progress = {}
        sample_batch.failed = [
            FailedTopic(topic="Topic E", error="Crash", stage="section")
        ]
        rm.save_batch_state(sample_batch)

        opts = rm.get_recovery_options()

        assert RecoveryAction.RESTART in opts.available_actions
        assert RecoveryAction.RETRY_ERRORS in opts.available_actions
        # CONTINUE should not be offered (no queued or in_progress).
        assert RecoveryAction.CONTINUE not in opts.available_actions

    def test_options_without_saved_state_raises(self, rm: RecoveryManager) -> None:
        """get_recovery_options without saved state should raise."""
        with pytest.raises(StateError):
            rm.get_recovery_options()


# ── Tests: topic state transitions ────────────────────────────────────


class TestTopicTransitions:
    """Tests for marking topics as started/completed/failed."""

    def test_mark_started(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """mark_topic_started should move topic from queued to in_progress."""
        rm.mark_topic_started(sample_batch, "Topic A", stage="concept")

        assert "Topic A" not in sample_batch.queued
        assert "Topic A" in sample_batch.in_progress
        assert sample_batch.in_progress["Topic A"].stage == "concept"

    def test_update_progress(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """update_topic_progress should update stage/section/attempt."""
        rm.mark_topic_started(sample_batch, "Topic A")
        rm.update_topic_progress(
            sample_batch, "Topic A", stage="section", section=2, attempt=3
        )

        progress = sample_batch.in_progress["Topic A"]
        assert progress.stage == "section"
        assert progress.section == 2
        assert progress.attempt == 3

    def test_mark_completed(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """mark_topic_completed should move topic from in_progress to completed."""
        rm.mark_topic_started(sample_batch, "Topic A")
        rm.mark_topic_completed(sample_batch, "Topic A")

        assert "Topic A" not in sample_batch.in_progress
        assert "Topic A" in sample_batch.completed

    def test_mark_failed(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """mark_topic_failed should move topic from in_progress to failed."""
        rm.mark_topic_started(sample_batch, "Topic A", stage="outline")
        rm.mark_topic_failed(
            sample_batch, "Topic A", error="API timeout", stage="outline", attempt=2
        )

        assert "Topic A" not in sample_batch.in_progress
        assert len(sample_batch.failed) == 1
        assert sample_batch.failed[0].topic == "Topic A"
        assert sample_batch.failed[0].error == "API timeout"
        assert sample_batch.failed[0].stage == "outline"
        assert sample_batch.failed[0].attempt == 2


# ── Tests: clear ──────────────────────────────────────────────────────


class TestClear:
    """Tests for clearing recovery state."""

    def test_clear_removes_file(
        self, rm: RecoveryManager, sample_batch: BatchState
    ) -> None:
        """clear() should delete the recovery state file."""
        rm.save_batch_state(sample_batch)
        assert rm.has_unfinished() is True

        rm.clear()
        assert rm.has_unfinished() is False

    def test_clear_when_no_file(self, rm: RecoveryManager) -> None:
        """clear() should not raise when no file exists."""
        rm.clear()  # Should not raise.
        assert rm.has_unfinished() is False


# ── Tests: simulate crash ─────────────────────────────────────────────


class TestSimulateCrash:
    """Tests that simulate a crash mid-batch and verify recovery."""

    def test_crash_and_recover(
        self, recovery_settings: Settings, sample_batch: BatchState
    ) -> None:
        """Simulate crash: save state, create new manager, verify options."""
        rm1 = RecoveryManager(recovery_settings)

        # Simulate some progress.
        rm1.mark_topic_started(sample_batch, "Topic A", stage="concept")
        rm1.mark_topic_completed(sample_batch, "Topic A")
        rm1.mark_topic_started(sample_batch, "Topic B", stage="section")
        rm1.update_topic_progress(
            sample_batch, "Topic B", stage="section", section=2, attempt=1
        )
        rm1.save_batch_state(sample_batch)

        # "Crash" — create a brand-new RecoveryManager.
        rm2 = RecoveryManager(recovery_settings)

        assert rm2.has_unfinished() is True

        opts = rm2.get_recovery_options()
        assert opts.completed_count == 1
        assert opts.in_progress_count == 1
        assert opts.queued_count == 3
        assert opts.failed_count == 0

        # Load full state and verify in_progress snapshot.
        loaded = rm2.load_batch_state()
        assert "Topic B" in loaded.in_progress
        assert loaded.in_progress["Topic B"].stage == "section"
        assert loaded.in_progress["Topic B"].section == 2

    def test_crash_with_mixed_states(
        self, recovery_settings: Settings
    ) -> None:
        """Crash with completed, in_progress, failed, and queued topics."""
        topics = [f"Topic {i}" for i in range(10)]
        batch = RecoveryManager.create_batch_state(topics=topics, language="de")

        rm1 = RecoveryManager(recovery_settings)

        # Completed: 0, 1, 2
        for i in range(3):
            rm1.mark_topic_started(batch, topics[i], stage="concept")
            rm1.mark_topic_completed(batch, topics[i])

        # In progress: 3, 4
        rm1.mark_topic_started(batch, topics[3], stage="outline")
        rm1.mark_topic_started(batch, topics[4], stage="section")
        rm1.update_topic_progress(
            batch, topics[4], stage="section", section=1, attempt=2
        )

        # Failed: 5
        rm1.mark_topic_started(batch, topics[5], stage="evaluate")
        rm1.mark_topic_failed(
            batch, topics[5], error="Low score after max attempts",
            stage="evaluate", attempt=5,
        )

        # Queued: 6, 7, 8, 9 (still in queue)
        rm1.save_batch_state(batch)

        # "Crash" — new manager.
        rm2 = RecoveryManager(recovery_settings)
        assert rm2.has_unfinished() is True

        opts = rm2.get_recovery_options()
        assert opts.completed_count == 3
        assert opts.in_progress_count == 2
        assert opts.failed_count == 1
        assert opts.queued_count == 4
        assert opts.total_topics == 10

        assert RecoveryAction.CONTINUE in opts.available_actions
        assert RecoveryAction.RESTART in opts.available_actions
        assert RecoveryAction.RETRY_ERRORS in opts.available_actions


# ── Tests: BatchState serialisation ───────────────────────────────────


class TestBatchStateSerialization:
    """Tests for BatchState pydantic round-trip."""

    def test_round_trip(self) -> None:
        """BatchState should survive JSON round-trip."""
        batch = BatchState(
            batch_id="test_123",
            total_topics=3,
            completed=["A"],
            in_progress={"B": TopicProgress(stage="section", section=1, attempt=2)},
            failed=[FailedTopic(topic="C", error="Timeout", stage="concept", attempt=1)],
            queued=[],
            started_at="2025-01-15T12:00:00Z",
            updated_at="2025-01-15T12:05:00Z",
            config_snapshot=ConfigSnapshot(
                language="de",
                target_words=5000,
                tone="suspenseful",
                model="claude-3",
                provider="anthropic",
            ),
        )

        json_str = batch.model_dump_json()
        restored = BatchState.model_validate_json(json_str)

        assert restored.batch_id == "test_123"
        assert restored.total_topics == 3
        assert restored.completed == ["A"]
        assert "B" in restored.in_progress
        assert restored.in_progress["B"].stage == "section"
        assert restored.in_progress["B"].section == 1
        assert restored.in_progress["B"].attempt == 2
        assert len(restored.failed) == 1
        assert restored.failed[0].topic == "C"
        assert restored.config_snapshot.language == "de"
        assert restored.config_snapshot.target_words == 5000
