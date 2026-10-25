"""dummyserver demonstrations of the outbound timeout-budget policy stack.

Three independently observable scenarios, split exactly along the required
exits:

1. ``wait_partition``: tighten *only* the pool queue wait; the read side is
   untouched and queue exhaustion still surfaces as ``EmptyPoolError``.
2. ``hop_clock=read_isolated``: a small ``total`` no longer eats into the
   read budget after connect, while the same numbers pin-style fail without
   the frame.
3. ``hop_clock=chain_remaining``: across one 3xx hop the remaining ``total``
   is carried over, so the second hop observably gets a smaller budget. Both
   the PoolManager cross-host recursion and the HTTPConnectionPool same-host
   recursion are exercised.
"""

from __future__ import annotations

import re
import socket
import threading
import time
import typing

import pytest

from dummyserver.socketserver import HAS_IPV6_AND_DNS
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import (
    HopClock,
    HTTPConnectionPool,
    PoolManager,
    WaitPartition,
    timeout_budget,
)
from urllib3.exceptions import EmptyPoolError, MaxRetryError, ReadTimeoutError
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout

OK = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

FIRST_HOP_DELAY = 0.4
SECOND_HOP_DELAY = 0.7
CHAIN_TOTAL = 1.0
SLOW_DELAY = 0.4
SMALL_TOTAL = 0.15
LARGE_READ = 2.0


def _listener_host() -> str:
    """Bare IP literal the dummy listener is actually bound to (avoids the
    localhost IPv4->IPv6 fallback being counted inside the connect clock)."""
    return "::1" if HAS_IPV6_AND_DNS else "127.0.0.1"


def _listener_url_host() -> str:
    """IP literal suitable for use inside an http:// URL (bracketed v6)."""
    return f"[{_listener_host()}]" if HAS_IPV6_AND_DNS else _listener_host()


def _accept_quit_aware(
    listener: socket.socket, quit_event: threading.Event
) -> socket.socket:
    listener.settimeout(0.05)
    while True:
        if quit_event.is_set():
            raise RuntimeError("server quit")
        try:
            return listener.accept()[0]
        except TimeoutError:
            continue


def _delayed_ok_handler(
    num_connections: int, delay: float, quit_event: threading.Event
) -> typing.Callable[[socket.socket], None]:
    def socket_handler(listener: socket.socket) -> None:
        for _ in range(num_connections):
            try:
                sock = _accept_quit_aware(listener, quit_event)
            except RuntimeError:
                return
            consume_socket(sock, quit_event=quit_event)
            if quit_event.wait(delay):
                sock.close()
                return
            sock.sendall(OK)
            sock.close()

    return socket_handler


class TestWaitPartitionDemos(SocketDummyServerTestCase):
    def test_tighten_only_pool_wait_not_read(self) -> None:
        self.start_basic_handler(num=3)
        host = _listener_host()
        with HTTPConnectionPool(
            host,
            self.port,
            maxsize=1,
            block=True,
            timeout=Timeout(connect=5, read=5),
            retries=False,
        ) as pool:
            # Hold the single connection outside the pool for the whole demo.
            held = pool.urlopen("GET", "/", preload_content=False)
            try:
                # Frame tightens the pool queue ceiling only; read/total are
                # not mentioned, so the socket ledgers keep the baseline.
                with timeout_budget(
                    pool=0.1,
                    wait_partition=WaitPartition.POOL_SEPARATE_CEILING,
                ):
                    start = time.monotonic()
                    with pytest.raises(EmptyPoolError):
                        pool.urlopen("GET", "/")
                    elapsed = time.monotonic() - start
                # Waited roughly the frame ceiling; never a ReadTimeoutError.
                assert 0.1 <= elapsed < 1.0, elapsed
            finally:
                held.release_conn()

            # Same frame policy on a free pool: reads are untouched and the
            # request succeeds normally.
            with timeout_budget(
                pool=0.1,
                wait_partition=WaitPartition.POOL_THEN_SOCKET,
            ):
                response = pool.urlopen("GET", "/")
            assert response.status == 200


class TestReadIsolatedDemos(SocketDummyServerTestCase):
    def test_read_isolated_keeps_full_read_budget(self) -> None:
        quit_event = threading.Event()
        self._start_server(
            _delayed_ok_handler(
                num_connections=2, delay=SLOW_DELAY, quit_event=quit_event
            ),
            quit_event=quit_event,
        )
        host = _listener_host()
        with HTTPConnectionPool(
            host,
            self.port,
            timeout=Timeout(connect=5, read=5),
            retries=0,
        ) as pool:
            pin_timeout = Timeout(connect=5, read=LARGE_READ, total=SMALL_TOTAL)

            # Control without a frame: the pin "total minus connect elapsed"
            # shrinks read well below the server delay -> ReadTimeoutError.
            with pytest.raises(MaxRetryError) as exc_info:
                pool.urlopen("GET", "/", timeout=pin_timeout)
            assert isinstance(exc_info.value.reason, ReadTimeoutError)

            # With READ_ISOLATED the same numbers decouple read from total:
            # the read side keeps LARGE_READ seconds and the slow response
            # arrives successfully.
            with timeout_budget(
                connect=5,
                read=LARGE_READ,
                total=SMALL_TOTAL,
                hop_clock=HopClock.READ_ISOLATED,
            ):
                start = time.monotonic()
                response = pool.urlopen("GET", "/")
                elapsed = time.monotonic() - start
            assert response.status == 200
            assert elapsed >= SLOW_DELAY


def _redirect_handler(
    location: bytes,
    quit_event: threading.Event,
    second_hop_delay: float | None = None,
) -> typing.Callable[[socket.socket], None]:
    """Hop 1: delayed 302. When second_hop_delay is given the same listener
    also serves the slow hop-2 200 (same-host recursion case)."""

    def socket_handler(listener: socket.socket) -> None:
        try:
            sock = _accept_quit_aware(listener, quit_event)
        except RuntimeError:
            return
        consume_socket(sock, quit_event=quit_event)
        if quit_event.wait(FIRST_HOP_DELAY):
            sock.close()
            return
        sock.sendall(
            b"HTTP/1.1 302 Found\r\n"
            b"Location: " + location + b"\r\n"
            b"Content-Length: 0\r\n"
            b"Connection: close\r\n"
            b"\r\n"
        )
        sock.close()
        if second_hop_delay is None:
            return
        # Hop 2: slower than the chain remainder but faster than a fresh
        # per-attempt total.
        try:
            sock = _accept_quit_aware(listener, quit_event)
        except RuntimeError:
            return
        consume_socket(sock, quit_event=quit_event)
        if quit_event.wait(second_hop_delay):
            sock.close()
            return
        sock.sendall(OK)
        sock.close()

    return socket_handler


class TestChainRemainingDemos(SocketDummyServerTestCase):
    def _redirect_retry(self) -> Retry:
        # Redirects allowed, connect/read errors terminal so the demo stays
        # at exactly two hops.
        return Retry(connect=0, read=0, redirect=5, status=0)

    def _assert_second_hop_budget(self, exc_info: pytest.ExceptionInfo) -> None:
        """The hop-2 read ceiling must be the carried remainder (~0.6s),
        not a fresh full CHAIN_TOTAL (1.0s)."""
        assert isinstance(exc_info.value.reason, ReadTimeoutError)
        match = re.search(r"read timeout=([\d.]+)", str(exc_info.value.reason))
        assert match is not None
        remaining = float(match.group(1))
        # Connect on loopback is negligible, so the carried remainder is
        # approximately CHAIN_TOTAL - FIRST_HOP_DELAY; give slack for
        # scheduling jitter.
        assert remaining < CHAIN_TOTAL - 0.2, remaining
        assert remaining > CHAIN_TOTAL - FIRST_HOP_DELAY - 0.2, remaining

    def _start_redirect_server(
        self, location: bytes, second_hop_delay: float | None = None
    ) -> None:
        quit_event = threading.Event()
        self._start_server(
            _redirect_handler(location, quit_event, second_hop_delay),
            quit_event=quit_event,
        )

    def test_per_attempt_gives_each_hop_a_full_total(self) -> None:
        url_host = _listener_url_host()
        self._start_redirect_server(b"/target", SECOND_HOP_DELAY)
        url = f"http://{url_host}:{self.port}/start"
        with PoolManager(
            timeout=Timeout(connect=5, read=5), retries=self._redirect_retry()
        ) as pm:
            # PER_ATTEMPT (pin): the second hop clones a fresh full total.
            with timeout_budget(
                total=CHAIN_TOTAL, hop_clock=HopClock.PER_ATTEMPT
            ):
                response = pm.urlopen("GET", url)
            assert response.status == 200

    def test_chain_remaining_cross_host_via_poolmanager(self) -> None:
        from dummyserver.socketserver import SocketServerThread

        host = _listener_host()
        url_host = _listener_url_host()
        # A genuinely distinct hop-2 origin: second listener, second port.
        quit2 = threading.Event()
        ready2 = threading.Event()
        server2 = SocketServerThread(
            _delayed_ok_handler(
                num_connections=1,
                delay=SECOND_HOP_DELAY,
                quit_event=quit2,
            ),
            host=host,
            ready_event=ready2,
            quit_event=quit2,
        )
        server2.start()
        try:
            assert ready2.wait(5.0)
            port2 = server2.port
            location = f"http://{url_host}:{port2}/target".encode()
            self._start_redirect_server(location)
            with PoolManager(
                timeout=Timeout(connect=5, read=5),
                retries=self._redirect_retry(),
            ) as pm:
                with pytest.raises(MaxRetryError) as exc_info:
                    with timeout_budget(
                        total=CHAIN_TOTAL,
                        hop_clock=HopClock.CHAIN_REMAINING,
                    ):
                        pm.urlopen(
                            "GET", f"http://{url_host}:{self.port}/start"
                        )
                # Second hop got ~CHAIN_TOTAL-FIRST_HOP_DELAY < 0.7s.
                self._assert_second_hop_budget(exc_info)
        finally:
            quit2.set()
            server2.join(6.0)

    def test_chain_remaining_same_host_via_connectionpool(self) -> None:
        host = _listener_host()
        self._start_redirect_server(b"/target", SECOND_HOP_DELAY)
        with HTTPConnectionPool(
            host,
            self.port,
            timeout=Timeout(connect=5, read=5),
            retries=self._redirect_retry(),
        ) as pool:
            with pytest.raises(MaxRetryError) as exc_info:
                with timeout_budget(
                    total=CHAIN_TOTAL,
                    hop_clock=HopClock.CHAIN_REMAINING,
                ):
                    pool.urlopen("GET", "/start")
            self._assert_second_hop_budget(exc_info)

    def test_chain_frame_popped_restores_default_policy(self) -> None:
        quit_event = threading.Event()
        self._start_server(
            _delayed_ok_handler(
                num_connections=1, delay=0.0, quit_event=quit_event
            ),
            quit_event=quit_event,
        )
        host = _listener_host()
        with HTTPConnectionPool(
            host,
            self.port,
            timeout=Timeout(connect=5, read=5),
            retries=0,
        ) as pool:
            with timeout_budget(
                total=SMALL_TOTAL,
                hop_clock=HopClock.CHAIN_REMAINING,
            ):
                pass
            # Stack is empty again: a normal request is not constrained by
            # the popped frame's tiny total.
            response = pool.urlopen("GET", "/")
            assert response.status == 200
