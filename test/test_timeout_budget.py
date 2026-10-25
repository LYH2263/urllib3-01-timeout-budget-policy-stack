from __future__ import annotations

import threading
import time
from queue import Empty
from unittest.mock import patch

import pytest

from urllib3 import (
    BudgetMerge,
    HopClock,
    TimeoutBudgetFrame,
    WaitPartition,
    timeout_budget,
)
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import EmptyPoolError
from urllib3.util import budget
from urllib3.util.timeout import Timeout, _DEFAULT_TIMEOUT

POOL_TIMEOUT = Timeout(connect=7, read=8, total=9)


def _parts(timeout_obj: Timeout) -> tuple[object, object, object]:
    return timeout_obj._connect, timeout_obj._read, timeout_obj.total


class TestEmptyStackParity:
    """Never-pushed deployments must keep the pin Timeout/pool_timeout behavior."""

    def test_resolve_matches_pin_get_timeout(self) -> None:
        pool = HTTPConnectionPool("example.com", timeout=POOL_TIMEOUT)
        # pool/caller default -> pool Timeout clone
        assert _parts(budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)) == (
            7,
            8,
            9,
        )
        # explicit Timeout -> clone of it
        assert _parts(
            budget.resolve_attempt_timeout(
                POOL_TIMEOUT, Timeout(total=3, connect=1, read=2)
            )
        ) == (1, 2, 3)
        # legacy float
        assert _parts(budget.resolve_attempt_timeout(POOL_TIMEOUT, 4)) == (
            4,
            4,
            None,
        )
        # explicit None -> infinite
        assert _parts(budget.resolve_attempt_timeout(POOL_TIMEOUT, None)) == (
            None,
            None,
            None,
        )
        # pool helper resolves through the same socket-side factory
        assert _parts(pool._get_timeout(_DEFAULT_TIMEOUT)) == (7, 8, 9)

    def test_resolve_clones_are_independent(self) -> None:
        raw = Timeout(1, 2, 3)
        first = budget.resolve_attempt_timeout(POOL_TIMEOUT, raw)
        second = budget.resolve_attempt_timeout(POOL_TIMEOUT, raw)
        assert first is not raw
        assert first is not second

    def test_pin_connect_and_read_formulas(self) -> None:
        timeout_obj = budget.resolve_attempt_timeout(
            POOL_TIMEOUT, Timeout(connect=10, read=10, total=0.2)
        )
        timeout_obj.start_connect()
        time.sleep(0.03)
        # connect is capped by total
        assert budget.resolve_connect_seconds(timeout_obj) == 0.2
        # read is total minus connect elapsed
        read_timeout = budget.resolve_read_seconds(timeout_obj)
        assert isinstance(read_timeout, float)
        assert 0 < read_timeout < 0.2

    def test_pool_timeout_passthrough(self) -> None:
        assert budget.resolve_pool_timeout(None) is None
        assert budget.resolve_pool_timeout(3.0) == 3.0

    def test_view_empty_stack(self) -> None:
        view = budget.get_effective_view(POOL_TIMEOUT, _DEFAULT_TIMEOUT, 1.5)
        assert view.depth == 0
        assert view.pool == 1.5
        assert view.connect == 7
        assert view.read == 8
        assert view.total == 9
        assert view.read_isolated is False
        assert view.chain_remaining is False

    def test_empty_urlopen_queue_uses_caller_pool_timeout(self) -> None:
        pool = HTTPConnectionPool("example.com", maxsize=1, block=True)
        with patch.object(pool.pool, "get", side_effect=Empty) as get_mock:
            # retries=False re-raises the raw queueing error (pin behaviour)
            with pytest.raises(EmptyPoolError):
                pool.urlopen("GET", "/", retries=False, pool_timeout=0.07)
        assert get_mock.call_args.kwargs["timeout"] == 0.07


class TestFrameLifecycle:
    def test_frame_validation(self) -> None:
        with pytest.raises(ValueError):
            TimeoutBudgetFrame(connect=0)
        with pytest.raises(ValueError):
            TimeoutBudgetFrame(total=-1)
        with pytest.raises(TypeError):
            TimeoutBudgetFrame(budget_merge="strict_min")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            TimeoutBudgetFrame(hop_clock="per_attempt")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            TimeoutBudgetFrame(wait_partition="socket_only")  # type: ignore[arg-type]

    def test_push_pop_restores_empty_stack(self) -> None:
        assert budget.current_frame() is None
        token = budget.push_frame(connect=1)
        try:
            assert budget.current_frame() is not None
        finally:
            budget.pop_frame(token)
        assert budget.current_frame() is None

    def test_context_manager_pops_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with timeout_budget(connect=1):
                assert budget.current_frame() is not None
                raise RuntimeError("boom")
        assert budget.current_frame() is None

    def test_inner_pop_returns_to_outer_policy_not_global(self) -> None:
        with timeout_budget(connect=5) as outer:
            assert budget.current_frame() is outer
            with timeout_budget(connect=2) as inner:
                assert budget.current_frame() is inner
                view = budget.get_effective_view(
                    POOL_TIMEOUT, _DEFAULT_TIMEOUT, None
                )
                assert view.connect == 2
                assert view.depth == 2
            assert budget.current_frame() is outer
            view = budget.get_effective_view(POOL_TIMEOUT, _DEFAULT_TIMEOUT, None)
            assert view.connect == 5
            assert view.depth == 1
        assert budget.current_frame() is None


class TestBudgetMerge:
    def test_strict_min_tightens(self) -> None:
        with timeout_budget(connect=5, read=5, total=8):
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (5, 5, 8)

    def test_strict_min_cannot_widen(self) -> None:
        with timeout_budget(connect=100, read=100, total=100):
            # baseline (7/8/9) wins over the loose frame numbers
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (7, 8, 9)

    def test_strict_min_unspecified_dimension_passes_through(self) -> None:
        with timeout_budget(connect=2):
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (2, 8, 9)

    def test_inherit_only_can_only_narrow(self) -> None:
        with timeout_budget(
            connect=100, total=1, budget_merge=BudgetMerge.INHERIT_ONLY
        ):
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (7, 8, 1)

    def test_active_frame_replaces_with_its_own_ceilings(self) -> None:
        # even wider numbers than the pool baseline are honored
        with timeout_budget(
            connect=30,
            read=40,
            total=50,
            budget_merge=BudgetMerge.ACTIVE_FRAME,
        ):
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (30, 40, 50)

    def test_active_frame_unspecified_dims_fall_back_to_baseline(self) -> None:
        with timeout_budget(
            read=40, budget_merge=BudgetMerge.ACTIVE_FRAME
        ):
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (7, 40, 9)

    def test_active_frame_discards_outer_frame_contributions(self) -> None:
        with timeout_budget(connect=2, total=2):
            with timeout_budget(
                connect=6, budget_merge=BudgetMerge.ACTIVE_FRAME
            ):
                # outer strict connect=2 is ignored; unspecified read/total
                # come from the pool baseline, not from the outer frame
                assert _parts(
                    budget.resolve_attempt_timeout(
                        POOL_TIMEOUT, _DEFAULT_TIMEOUT
                    )
                ) == (6, 8, 9)


class TestWaitPartition:
    def test_socket_only_does_not_order_pool_timeout(self) -> None:
        with timeout_budget(pool=0.01):
            assert budget.resolve_pool_timeout(None) is None
            assert budget.resolve_pool_timeout(2.0) == 2.0

    def test_pool_then_socket_caps_queue_but_keeps_ledgers_separate(self) -> None:
        with timeout_budget(
            pool=0.05, wait_partition=WaitPartition.POOL_THEN_SOCKET
        ):
            # caller None (block forever) -> frame ceiling
            assert budget.resolve_pool_timeout(None) == 0.05
            # stricter caller value wins
            assert budget.resolve_pool_timeout(0.02) == 0.02
            # the pool number never enters the socket Timeout ledger
            assert _parts(
                budget.resolve_attempt_timeout(POOL_TIMEOUT, _DEFAULT_TIMEOUT)
            ) == (7, 8, 9)

    def test_pool_separate_ceiling_is_independent(self) -> None:
        with timeout_budget(
            pool=0.03,
            wait_partition=WaitPartition.POOL_SEPARATE_CEILING,
        ):
            assert budget.resolve_pool_timeout(None) == 0.03
            assert budget.resolve_pool_timeout(0.5) == 0.03
            # it must not be folded into Timeout.total
            view = budget.get_effective_view(
                POOL_TIMEOUT, _DEFAULT_TIMEOUT, None
            )
            assert view.pool == 0.03
            assert view.total == 9

    def test_nested_pool_frames_take_strict_min(self) -> None:
        with timeout_budget(
            pool=0.1, wait_partition=WaitPartition.POOL_THEN_SOCKET
        ):
            with timeout_budget(
                pool=0.02,
                wait_partition=WaitPartition.POOL_SEPARATE_CEILING,
            ):
                assert budget.resolve_pool_timeout(None) == 0.02
            assert budget.resolve_pool_timeout(None) == 0.1

    def test_queue_exit_uses_frame_ceiling_via_urlopen(self) -> None:
        pool = HTTPConnectionPool("example.com", maxsize=1, block=True)
        with timeout_budget(
            pool=0.02, wait_partition=WaitPartition.POOL_THEN_SOCKET
        ):
            with patch.object(pool.pool, "get", side_effect=Empty) as get_mock:
                with pytest.raises(EmptyPoolError):
                    pool.urlopen("GET", "/", retries=False)
        assert get_mock.call_args.kwargs["timeout"] == 0.02

    def test_queue_exit_real_blocking_wait(self) -> None:
        pool = HTTPConnectionPool("example.com", maxsize=1, block=True)
        # Remove the single reusable-connection token so urlopen has to queue.
        pool.pool.get_nowait()
        start = time.monotonic()
        with timeout_budget(
            pool=0.03, wait_partition=WaitPartition.POOL_SEPARATE_CEILING
        ):
            with pytest.raises(EmptyPoolError):
                pool.urlopen("GET", "/", retries=False)
        elapsed = time.monotonic() - start
        assert 0.03 <= elapsed < 1.0, elapsed


class TestHopClock:
    # Generous baseline without its own total, so the frame totals govern.
    CHAIN_BASELINE = Timeout(connect=30, read=30, total=None)

    def test_per_attempt_gives_full_total_every_time(self) -> None:
        with timeout_budget(total=10) as frame:
            first = budget.resolve_attempt_timeout(
                POOL_TIMEOUT, _DEFAULT_TIMEOUT
            )
            first.start_connect()
            time.sleep(0.02)
            budget.note_attempt_elapsed(first)
            second = budget.resolve_attempt_timeout(
                POOL_TIMEOUT, _DEFAULT_TIMEOUT
            )
            assert frame.chain_consumed == 0.0
            assert second.total == 9  # strict_min with baseline total 9

    def test_chain_remaining_carries_consumed_total(self) -> None:
        with timeout_budget(
            total=10, hop_clock=HopClock.CHAIN_REMAINING
        ) as frame:
            first = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            assert first.total == 10
            first.start_connect()
            time.sleep(0.05)
            budget.note_attempt_elapsed(first)
            assert frame.chain_consumed >= 0.035
            second = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            assert isinstance(second.total, float)
            assert second.total < 10
            assert second.total <= 10 - frame.chain_consumed + 0.001

    def test_chain_remaining_excludes_retry_sleep_wall_clock(self) -> None:
        with timeout_budget(
            total=10, hop_clock=HopClock.CHAIN_REMAINING
        ) as frame:
            first = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            first.start_connect()
            time.sleep(0.02)
            # account the attempt BEFORE the (simulated) retry backoff
            budget.note_attempt_elapsed(first)
            consumed = frame.chain_consumed
            time.sleep(0.1)  # Retry.sleep / Retry-After wall clock
            second = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            # the 0.1s backoff must not show up in the ledger
            assert isinstance(second.total, float)
            assert second.total == pytest.approx(10 - consumed, abs=0.02)

    def test_chain_remaining_exhausted_budget_is_expired(self) -> None:
        with timeout_budget(
            total=0.05, hop_clock=HopClock.CHAIN_REMAINING
        ):
            first = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            first.start_connect()
            time.sleep(0.08)
            budget.note_attempt_elapsed(first)
            second = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            assert second.total == 0
            assert budget.resolve_connect_seconds(second) == 0

    def test_unstarted_attempt_records_nothing(self) -> None:
        with timeout_budget(
            total=10, hop_clock=HopClock.CHAIN_REMAINING
        ) as frame:
            fresh = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            budget.note_attempt_elapsed(fresh)
            assert frame.chain_consumed == 0.0

    def test_read_isolated_does_not_deduct_connect_from_total(self) -> None:
        with timeout_budget(
            connect=10,
            read=10,
            total=0.2,
            hop_clock=HopClock.READ_ISOLATED,
        ):
            timeout_obj = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            assert timeout_obj.total == 0.2
            timeout_obj.start_connect()
            time.sleep(0.05)
            assert budget.resolve_connect_seconds(timeout_obj) == 10
            assert budget.resolve_read_seconds(timeout_obj) == 10

    def test_read_isolated_control_without_frame_uses_pin_formula(self) -> None:
        timeout_obj = budget.resolve_attempt_timeout(
            POOL_TIMEOUT, Timeout(connect=10, read=10, total=0.2)
        )
        timeout_obj.start_connect()
        time.sleep(0.05)
        assert budget.resolve_connect_seconds(timeout_obj) == 0.2
        read_timeout = budget.resolve_read_seconds(timeout_obj)
        assert isinstance(read_timeout, float)
        assert read_timeout < 0.2

    def test_read_isolated_does_not_apply_to_pool_wait(self) -> None:
        with timeout_budget(
            pool=0.04,
            hop_clock=HopClock.READ_ISOLATED,
            wait_partition=WaitPartition.POOL_THEN_SOCKET,
        ):
            assert budget.resolve_pool_timeout(None) == 0.04
            view = budget.get_effective_view(
                POOL_TIMEOUT, _DEFAULT_TIMEOUT, None
            )
            assert view.read_isolated is True
            assert view.pool == 0.04

    def test_outer_chain_and_inner_strict_frame_compose(self) -> None:
        with timeout_budget(
            total=10, hop_clock=HopClock.CHAIN_REMAINING
        ):
            first = budget.resolve_attempt_timeout(
                self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
            )
            first.start_connect()
            time.sleep(0.03)
            budget.note_attempt_elapsed(first)
            with timeout_budget(connect=2):
                second = budget.resolve_attempt_timeout(
                    self.CHAIN_BASELINE, _DEFAULT_TIMEOUT
                )
                assert second._connect == 2
                assert isinstance(second.total, float)
                assert second.total < 10


class TestConcurrentStacks:
    def test_concurrent_urlopen_chains_do_not_share_frames(self) -> None:
        barrier = threading.Barrier(2)
        # baseline without a total so the frame totals are directly visible
        baseline = Timeout(connect=10, read=10, total=None)
        results: dict[str, object] = {}

        def worker(name: str, total: float) -> None:
            with timeout_budget(
                total=total, hop_clock=HopClock.CHAIN_REMAINING
            ) as frame:
                barrier.wait()
                view = budget.get_effective_view(baseline, _DEFAULT_TIMEOUT, None)
                frames = budget._frames()
                results[name] = (frame, view.total, len(frames), frames)

        t1 = threading.Thread(target=worker, args=("a", 11))
        t2 = threading.Thread(target=worker, args=("b", 12))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        frame_a, total_a, depth_a, frames_a = results["a"]
        frame_b, total_b, depth_b, frames_b = results["b"]
        assert depth_a == 1 and depth_b == 1
        assert frame_a is not frame_b
        assert frames_a[0] is frame_a and frames_b[0] is frame_b
        # each chain observes its own frame total, never the other thread's
        assert total_a == 11 and total_b == 12
        # the main thread never inherited either frame
        assert budget.current_frame() is None
        assert budget._frames() == ()
