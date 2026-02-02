"""Tests for batching types (RequestStatus, Priority, BatchRequest)."""

from mlx_manager.mlx_server.services.batching import (
    BatchRequest,
    Priority,
    RequestStatus,
)


class TestRequestStatus:
    """Tests for RequestStatus enum."""

    def test_has_all_states(self) -> None:
        """RequestStatus should have all 5 lifecycle states."""
        states = [s.name for s in RequestStatus]
        assert "WAITING" in states
        assert "PREFILLING" in states
        assert "RUNNING" in states
        assert "COMPLETED" in states
        assert "CANCELLED" in states
        assert len(states) == 5

    def test_ordering(self) -> None:
        """States should be ordered by lifecycle progression."""
        assert RequestStatus.WAITING < RequestStatus.PREFILLING
        assert RequestStatus.PREFILLING < RequestStatus.RUNNING
        assert RequestStatus.RUNNING < RequestStatus.COMPLETED


class TestPriority:
    """Tests for Priority enum."""

    def test_ordering_high_beats_low(self) -> None:
        """HIGH priority should have lower numeric value than LOW."""
        assert Priority.HIGH.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.LOW.value

    def test_numeric_values(self) -> None:
        """Priority values should match expected order."""
        assert Priority.HIGH.value == 0
        assert Priority.NORMAL.value == 1
        assert Priority.LOW.value == 2


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_creation_with_defaults(self) -> None:
        """BatchRequest should be creatable with minimal required fields."""
        request = BatchRequest(
            request_id="test-123",
            model_id="mlx-community/test-model",
            prompt_tokens=[1, 2, 3, 4, 5],
            max_tokens=100,
        )

        assert request.request_id == "test-123"
        assert request.model_id == "mlx-community/test-model"
        assert request.prompt_tokens == [1, 2, 3, 4, 5]
        assert request.max_tokens == 100
        assert request.priority == Priority.NORMAL  # default
        assert request.status == RequestStatus.WAITING  # default
        assert request.generated_tokens == []  # default factory
        assert request.started_at is None

    def test_base_priority_returns_priority_value(self) -> None:
        """base_priority property should return priority enum value."""
        for priority in Priority:
            request = BatchRequest(
                request_id="test",
                model_id="model",
                prompt_tokens=[1],
                max_tokens=10,
                priority=priority,
            )
            assert request.base_priority == priority.value

    def test_is_complete_when_max_tokens_reached(self) -> None:
        """is_complete should be True when generated_tokens reaches max_tokens."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=3,
        )

        assert not request.is_complete

        request.generated_tokens = [10, 20]
        assert not request.is_complete

        request.generated_tokens = [10, 20, 30]
        assert request.is_complete

    def test_is_complete_when_status_completed(self) -> None:
        """is_complete should be True when status is COMPLETED."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        request.status = RequestStatus.COMPLETED
        assert request.is_complete

    def test_is_complete_when_status_cancelled(self) -> None:
        """is_complete should be True when status is CANCELLED."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        request.status = RequestStatus.CANCELLED
        assert request.is_complete

    def test_is_complete_false_for_running(self) -> None:
        """is_complete should be False for RUNNING status with tokens remaining."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        request.status = RequestStatus.RUNNING
        assert not request.is_complete

    def test_effective_tokens(self) -> None:
        """effective_tokens should sum prompt and generated tokens."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1, 2, 3, 4, 5],  # 5 tokens
            max_tokens=100,
        )

        assert request.effective_tokens == 5

        request.generated_tokens = [10, 20, 30]  # 3 tokens
        assert request.effective_tokens == 8

    def test_add_token(self) -> None:
        """add_token should append to generated_tokens."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        request.add_token(42)
        assert request.generated_tokens == [42]

        request.add_token(43)
        assert request.generated_tokens == [42, 43]

    def test_state_transitions(self) -> None:
        """State transition methods should update status correctly."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        assert request.status == RequestStatus.WAITING

        request.mark_prefilling()
        assert request.status == RequestStatus.PREFILLING
        assert request.started_at is not None

        request.mark_running()
        assert request.status == RequestStatus.RUNNING

        request.mark_completed()
        assert request.status == RequestStatus.COMPLETED

    def test_mark_cancelled(self) -> None:
        """mark_cancelled should set status to CANCELLED."""
        request = BatchRequest(
            request_id="test",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        request.mark_cancelled()
        assert request.status == RequestStatus.CANCELLED

    def test_each_request_has_own_generated_tokens_list(self) -> None:
        """Each request should have its own generated_tokens list (not shared)."""
        r1 = BatchRequest(
            request_id="r1",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )
        r2 = BatchRequest(
            request_id="r2",
            model_id="model",
            prompt_tokens=[1],
            max_tokens=100,
        )

        r1.add_token(10)
        assert r1.generated_tokens == [10]
        assert r2.generated_tokens == []  # Should not be affected
