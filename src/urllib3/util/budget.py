"""Outbound attempt-level timeout budget policy stack.

This module implements a *budget policy stack* that is deliberately
independent of :class:`~urllib3.util.retry.Retry` and of the default
:class:`~urllib3.util.timeout.Timeout` fields kept on ``PoolManager`` /
``ConnectionPool``.

A stack is bound to one user-visible ``urlopen()`` call chain (including its
internal 3xx redirect and error-retry recursion).  It is stored in a
:mod:`contextvars` variable, so two concurrent ``urlopen()`` calls sharing the
same pool each see their own stack.  When the stack is empty every observation
is byte-for-byte equivalent to relying solely on ``Timeout`` /
``pool_timeout`` / ``Retry``.

Each stack frame independently expresses three policies:

``budget_merge``
    How the frame's numbers relate to the outer frames and to the pool/caller
    baseline. ``STRICT_MIN`` only ever tightens (numeric minimum),
    ``ACTIVE_FRAME`` replaces the view with the frame's own ceilings, and
    ``INHERIT_ONLY`` lets an inner frame narrow but never widen the budget
    already in effect.

``hop_clock``
    How the attempt clock behaves across the redirect/retry hops of one call
    chain. ``PER_ATTEMPT`` is the pin behaviour (every hop clones a fresh
    ``Timeout`` and re-runs ``start_connect()``, ``total`` is never carried
    across hops). ``CHAIN_REMAINING`` carries the remaining ``total`` across
    recursive hops: the next hop must already see the time consumed by prior
    hops before its own ``start_connect()``. ``READ_ISOLATED`` stops deducting
    connect elapsed from ``total`` on the read side; connect and read are each
    bounded by their own merged number (the rule never applies to pool
    queueing).

``wait_partition``
    How pool queue waiting relates to the socket budget. ``SOCKET_ONLY``
    leaves ``pool_timeout`` governed solely by the caller/pool baseline.
    ``POOL_THEN_SOCKET`` makes the frame's pool-wait ceiling and its socket
    budget both effective while keeping the two ledgers separate.
    ``POOL_SEPARATE_CEILING`` provides the pool-wait ceiling as a standalone
    ceiling that is never folded into ``Timeout.total`` and never faked with a
    single ``socket.settimeout()``.

The five outbound exits (pool queue wait, HTTP CONNECT tunnel handshake,
origin connect, origin read, recursive next hop) are served by *separate*
read functions; there is deliberately no single helper returning one value
for all of them.
"""

from __future__ import annotations

import contextlib
import enum
import typing
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from .timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT, Timeout

_TYPE_COMPONENT = typing.Union[float, _TYPE_TIMEOUT, None]
_TYPE_STACK = typing.Tuple["TimeoutBudgetFrame", ...]


class BudgetMerge(enum.Enum):
    """How a frame's numbers merge with the outer frames and baseline."""

    STRICT_MIN = "strict_min"
    ACTIVE_FRAME = "active_frame"
    INHERIT_ONLY = "inherit_only"


class HopClock(enum.Enum):
    """How the attempt clock behaves across hops of one call chain."""

    PER_ATTEMPT = "per_attempt"
    CHAIN_REMAINING = "chain_remaining"
    READ_ISOLATED = "read_isolated"


class WaitPartition(enum.Enum):
    """How pool queue waiting is accounted relative to the socket budget."""

    SOCKET_ONLY = "socket_only"
    POOL_THEN_SOCKET = "pool_then_socket"
    POOL_SEPARATE_CEILING = "pool_separate_ceiling"


def _validate_ceiling(value: float | None, name: str) -> float | None:
    # Reuse Timeout's numeric validation (positive numbers / None only).
    validated = Timeout._validate_timeout(value, name)
    if validated is None or validated is _DEFAULT_TIMEOUT:
        return None
    return validated


@dataclass
class TimeoutBudgetFrame:
    """One frame of the outbound attempt-level timeout budget stack.

    ``connect``/``read``/``total``/``pool`` are positive second ceilings;
    ``None`` means the frame does not mention that dimension.
    ``chain_consumed`` accumulates the connect-to-response time of attempts
    completed while the frame was active, but only when ``hop_clock`` is
    :attr:`HopClock.CHAIN_REMAINING`.  ``Retry.sleep()`` /
    ``sleep_for_retry()`` wall clock (including ``Retry-After``) is never
    added here and therefore stays outside the ``Timeout`` ledger.
    """

    connect: float | None = None
    read: float | None = None
    total: float | None = None
    pool: float | None = None
    budget_merge: BudgetMerge = BudgetMerge.STRICT_MIN
    hop_clock: HopClock = HopClock.PER_ATTEMPT
    wait_partition: WaitPartition = WaitPartition.SOCKET_ONLY
    chain_consumed: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.connect = _validate_ceiling(self.connect, "connect")
        self.read = _validate_ceiling(self.read, "read")
        self.total = _validate_ceiling(self.total, "total")
        self.pool = _validate_ceiling(self.pool, "pool")
        if not isinstance(self.budget_merge, BudgetMerge):
            raise TypeError("budget_merge must be a BudgetMerge value")
        if not isinstance(self.hop_clock, HopClock):
            raise TypeError("hop_clock must be a HopClock value")
        if not isinstance(self.wait_partition, WaitPartition):
            raise TypeError("wait_partition must be a WaitPartition value")


#: Stack for the current context. An empty tuple means "no frame pushed";
#: there is deliberately no process-wide default frame singleton.
_budget_stack: ContextVar[_TYPE_STACK] = ContextVar(
    "urllib3_timeout_budget_stack", default=()
)


def _frames() -> _TYPE_STACK:
    return _budget_stack.get()


def push_frame(
    connect: float | None = None,
    read: float | None = None,
    total: float | None = None,
    pool: float | None = None,
    budget_merge: BudgetMerge = BudgetMerge.STRICT_MIN,
    hop_clock: HopClock = HopClock.PER_ATTEMPT,
    wait_partition: WaitPartition = WaitPartition.SOCKET_ONLY,
) -> Token[_TYPE_STACK]:
    """Push a budget frame onto the current context's stack.

    Pair with :func:`pop_frame` (or prefer the :func:`timeout_budget` context
    manager, which also pops on exception paths).
    """
    frame = TimeoutBudgetFrame(
        connect=connect,
        read=read,
        total=total,
        pool=None if pool is None else float(pool),
        budget_merge=budget_merge,
        hop_clock=hop_clock,
        wait_partition=wait_partition,
    )
    return _budget_stack.set(_frames() + (frame,))


def pop_frame(token: Token[_TYPE_STACK]) -> None:
    """Pop a frame previously pushed with :func:`push_frame`."""
    _budget_stack.reset(token)


@contextlib.contextmanager
def timeout_budget(
    connect: float | None = None,
    read: float | None = None,
    total: float | None = None,
    pool: float | None = None,
    budget_merge: BudgetMerge = BudgetMerge.STRICT_MIN,
    hop_clock: HopClock = HopClock.PER_ATTEMPT,
    wait_partition: WaitPartition = WaitPartition.SOCKET_ONLY,
) -> typing.Generator[TimeoutBudgetFrame, None, None]:
    """Context manager that pushes a frame for its duration and pops it on exit.

    The frame is removed on every exit, including exceptions. Nested uses form
    a stack: after the inner frame is popped, the outer user policy is visible
    again -- execution never falls back to a process-wide global frame.
    """
    token = push_frame(
        connect=connect,
        read=read,
        total=total,
        pool=pool,
        budget_merge=budget_merge,
        hop_clock=hop_clock,
        wait_partition=wait_partition,
    )
    frame = _frames()[-1]
    try:
        yield frame
    finally:
        pop_frame(token)


def current_frame() -> TimeoutBudgetFrame | None:
    """Return the top frame of the current context stack, or ``None``."""
    frames = _frames()
    return frames[-1] if frames else None


def _min_component(
    a: _TYPE_COMPONENT, b: _TYPE_COMPONENT
) -> _TYPE_COMPONENT:
    """Dimension-wise minimum that preserves the ``Timeout`` sentinels.

    A concrete number always wins over ``None`` (infinite) and the
    ``_DEFAULT_TIMEOUT`` sentinel (system default). Between two non-numbers,
    the sentinel is kept (system default is stricter than explicit infinity).
    """
    numbers = [
        value
        for value in (a, b)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if numbers:
        return min(numbers)
    if a is _DEFAULT_TIMEOUT or b is _DEFAULT_TIMEOUT:
        return _DEFAULT_TIMEOUT
    return None


def _baseline_timeout(
    pool_default: Timeout, raw_timeout: _TYPE_TIMEOUT
) -> Timeout:
    """Mirror the pin ``HTTPConnectionPool._get_timeout`` resolution."""
    if raw_timeout is _DEFAULT_TIMEOUT:
        return pool_default.clone()
    if isinstance(raw_timeout, Timeout):
        return raw_timeout.clone()
    # User passed an int/float (legacy) or None.
    return Timeout.from_float(raw_timeout)


def _fold_socket_timeout(
    pool_default: Timeout, raw_timeout: _TYPE_TIMEOUT
) -> Timeout:
    base = _baseline_timeout(pool_default, raw_timeout)
    frames = _frames()
    if not frames:
        return base

    connect_value: _TYPE_COMPONENT = base._connect
    read_value: _TYPE_COMPONENT = base._read
    total_value: _TYPE_COMPONENT = base.total

    for frame in frames:
        if frame.budget_merge is BudgetMerge.ACTIVE_FRAME:
            # Only the frame's own ceilings count; dimensions it does not
            # mention fall back to the pool/caller baseline, discarding any
            # contribution from outer frames.
            connect_value = (
                frame.connect if frame.connect is not None else base._connect
            )
            read_value = frame.read if frame.read is not None else base._read
            total_value = frame.total if frame.total is not None else base.total
        else:
            # STRICT_MIN and INHERIT_ONLY: only narrow, never widen.
            if frame.connect is not None:
                connect_value = _min_component(connect_value, frame.connect)
            if frame.read is not None:
                read_value = _min_component(read_value, frame.read)
            if frame.total is not None:
                total_value = _min_component(total_value, frame.total)

    # CHAIN_REMAINING debt always applies as a hard minimum on top of the
    # configuration fold: a hop may never receive a "full" total while prior
    # hops of the same user call chain have already consumed part of it.
    for frame in frames:
        if (
            frame.hop_clock is HopClock.CHAIN_REMAINING
            and frame.total is not None
        ):
            remaining = frame.total - frame.chain_consumed
            total_value = _min_component(total_value, remaining)

    # An exhausted (or overspent) chain budget expires immediately instead of
    # feeding a negative number into socket.settimeout.
    if (
        isinstance(total_value, (int, float))
        and not isinstance(total_value, bool)
        and total_value < 0
    ):
        total_value = 0

    merged = Timeout(connect=_DEFAULT_TIMEOUT, read=_DEFAULT_TIMEOUT, total=None)
    # Assign directly: chain remaining may legitimately be zero (or negative)
    # once a budget is exhausted, which the Timeout constructor rejects.
    merged._connect = connect_value
    merged._read = read_value
    merged.total = total_value
    return merged


def resolve_attempt_timeout(
    pool_default: Timeout, raw_timeout: _TYPE_TIMEOUT
) -> Timeout:
    """Socket-side factory: the effective :class:`Timeout` for one attempt.

    Exit points served by this view are the origin connect, the HTTP CONNECT
    tunnel handshake (connect side only), the origin read and the recursive
    next hop.  With an empty stack this is identical to
    ``HTTPConnectionPool._get_timeout``.
    """
    return _fold_socket_timeout(pool_default, raw_timeout)


def resolve_pool_timeout(caller_pool_timeout: float | None) -> float | None:
    """Pool-queue-side factory (kept separate from the socket Timeout factory).

    Returns the effective number of seconds to wait in ``_get_conn``'s
    ``queue.get()``.  The result is never folded into ``Timeout.total`` and is
    never applied through ``socket.settimeout``: the queue ledger and the
    socket ledger stay separate.  With an empty stack (or only
    ``SOCKET_ONLY`` frames) the caller value is returned unchanged.
    """
    ceiling: float | None = None
    for frame in _frames():
        if (
            frame.wait_partition is not WaitPartition.SOCKET_ONLY
            and frame.pool is not None
        ):
            ceiling = frame.pool if ceiling is None else min(ceiling, frame.pool)
    if ceiling is None:
        return caller_pool_timeout
    if caller_pool_timeout is None:
        return ceiling
    return min(caller_pool_timeout, ceiling)


def read_isolation_active() -> bool:
    """Whether read/connect decoupling is in effect for this attempt.

    True when any frame on the stack selects :attr:`HopClock.READ_ISOLATED`;
    an inner frame cannot re-couple what an outer frame decoupled.
    """
    return any(
        frame.hop_clock is HopClock.READ_ISOLATED for frame in _frames()
    )


def chain_remaining_active() -> bool:
    """Whether any active frame carries total across recursive hops."""
    return any(
        frame.hop_clock is HopClock.CHAIN_REMAINING for frame in _frames()
    )


def resolve_connect_seconds(timeout_obj: Timeout) -> float | None:
    """Seconds for the connect-side socket timeout.

    Used for the origin connect *and* the HTTP CONNECT tunnel handshake in
    ``_prepare_proxy`` / ``_validate_conn``: those exits only ever see the
    connect-side merged view, never the read-side exemption.

    With an empty stack this is exactly
    ``Timeout.resolve_default_timeout(timeout_obj.connect_timeout)``.
    """
    if read_isolation_active():
        # connect is bounded by its own merged number; total must not couple
        # connect to read inside a read-isolated frame.
        return Timeout.resolve_default_timeout(timeout_obj._connect)
    return Timeout.resolve_default_timeout(timeout_obj.connect_timeout)


def resolve_read_seconds(timeout_obj: Timeout) -> float | None:
    """Seconds for the read-side socket timeout of one attempt.

    With an empty stack this is exactly ``timeout_obj.read_timeout`` (the pin
    formula deducting connect elapsed from ``total``). Under
    :attr:`HopClock.READ_ISOLATED` the merged read number is returned as-is
    and connect elapsed is not subtracted from total. Never consulted for
    pool queue waits or the CONNECT tunnel.
    """
    if read_isolation_active():
        return Timeout.resolve_default_timeout(timeout_obj._read)
    return timeout_obj.read_timeout


def note_attempt_elapsed(timeout_obj: Timeout) -> None:
    """Distribute one finished attempt's elapsed time into CHAIN frames.

    Called when an outbound attempt ends (response headers received or error
    raised), *before* any redirect/retry sleep, so ``Retry.sleep()`` and
    ``Retry-After`` wall clock stay outside the ledger. The elapsed time is
    measured from the attempt's own ``start_connect()`` exactly like the pin
    ``total`` bookkeeping.
    """
    if timeout_obj._start_connect is None:
        return
    duration = timeout_obj.get_connect_duration()
    for frame in _frames():
        if frame.hop_clock is HopClock.CHAIN_REMAINING:
            frame.chain_consumed += duration


@dataclass(frozen=True)
class EffectiveTimeoutBudget:
    """Read-only effective view for one ``urlopen`` call."""

    connect: float | None
    read: float | None
    total: _TYPE_COMPONENT
    pool: float | None
    read_isolated: bool
    chain_remaining: bool
    depth: int


def get_effective_view(
    pool_default: Timeout,
    raw_timeout: _TYPE_TIMEOUT,
    caller_pool_timeout: float | None,
) -> EffectiveTimeoutBudget:
    """Effective view after merging the stack with the call baselines.

    Merges the top-of-stack policy with the pool default :class:`Timeout` and
    the ``timeout`` / ``pool_timeout`` arguments passed to one ``urlopen``.
    Observing this view with an empty stack is equivalent to inspecting the
    pin-resolved ``Timeout`` / ``pool_timeout`` values directly.
    """
    merged = _fold_socket_timeout(pool_default, raw_timeout)
    frames = _frames()
    isolated = any(
        frame.hop_clock is HopClock.READ_ISOLATED for frame in frames
    )
    # The throwaway clone never had its clock started; start it just for the
    # pre-attempt observation so a total-only pin Timeout can resolve a read
    # value instead of raising TimeoutStateError.
    if merged._start_connect is None:
        merged.start_connect()
    connect_seconds = resolve_connect_seconds(merged)
    read_seconds = resolve_read_seconds(merged)
    return EffectiveTimeoutBudget(
        connect=connect_seconds,
        read=read_seconds,
        total=merged.total,
        pool=resolve_pool_timeout(caller_pool_timeout),
        read_isolated=isolated,
        chain_remaining=any(
            frame.hop_clock is HopClock.CHAIN_REMAINING for frame in frames
        ),
        depth=len(frames),
    )
