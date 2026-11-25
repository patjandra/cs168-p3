"""Unit tests for protocol.py: packet format, round-robin order, receiver.

These drive MySender/MyReceiver directly with a fake simulator, so they run
in milliseconds and need no ns-3 build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import protocol as P  # noqa: E402
from transportlab.api import AppMessage  # noqa: E402


class FakeSim:
    """Stands in for transportlab._tlsim: records sends, holds timers."""

    def __init__(self):
        self.t = 0.0
        self.sent = []      # (bytes, (ip, port))
        self.timers = {}    # name -> fire time

    def now(self):
        return self.t

    def send(self, data, ip, port):
        self.sent.append((data, (ip, port)))
        return True

    def set_timer(self, name, delay):
        self.timers[name] = self.t + delay

    def cancel_timer(self, name):
        self.timers.pop(name, None)

    def timer_pending(self, name):
        return name in self.timers

    def deliver(self, msg_id, payload):
        pass


def addr_of(i):
    return (f"10.0.0.{i}", 9000)


def make_sender(tmp_path, name="sender3", receivers=range(9)):
    sim = FakeSim()
    peers = {f"receiver{i}": addr_of(i) for i in receivers}
    s = P.MySender(name, "0.0.0.0", 0, peers, 0.0, 1, tmp_path, _sim=sim, schedule=[])
    s.on_start()
    return s, sim


def make_receiver(tmp_path, expected):
    sim = FakeSim()
    peers = {f"sender{i}": addr_of(i) for i in range(9)}
    r = P.MyReceiver("receiver0", "0.0.0.0", 0, peers, 0.0, 1, tmp_path, _sim=sim,
                     expected=expected)
    r.on_start()
    return r, sim


def fire(endpoint, sim, name):
    assert sim.timers.pop(name) == sim.t, f"{name} not due now"
    endpoint.on_timer(name)


def msg(msg_id, dst, size=1400):
    return AppMessage(msg_id=msg_id, dst=dst, size=size, deadline_s=None, offered_s=0.0)


def sent_dsts(sim):
    by_addr = {addr_of(i): f"receiver{i}" for i in range(9)}
    return [by_addr[addr] for _, addr in sim.sent]


# ---- packet format ---------------------------------------------------------

def test_header_is_six_bytes():
    assert P.HEADER.size == 6
    assert P.MAX_PAYLOAD == 1466


@pytest.mark.parametrize("payload", [b"", b"x" * 1400, b"y" * P.MAX_PAYLOAD])
def test_data_round_trips(payload):
    pkt = P.encode_data(123456, payload, send_no=3)
    assert len(pkt) == 6 + len(payload) <= P.MAX_DATAGRAM
    assert P.decode(pkt) == (P.TYPE_DATA, 3, 123456, payload)


def test_ack_round_trips():
    assert P.decode(P.encode_ack(2**32 - 1, 7)) == (P.TYPE_ACK, 7, 2**32 - 1, b"")


def test_send_number_saturates_instead_of_overflowing():
    assert P.decode(P.encode_data(1, b"", send_no=1000)).send_no == P.MAX_SEND_NO


@pytest.mark.parametrize("junk", [b"", b"\x00\x01\x00\x00\x01", b"\x07\x01\x00\x00\x00\x01",
                                  P.encode_ack(5, 1) + b"extra"])
def test_decode_rejects_junk(junk):
    assert P.decode(junk) is None


# ---- round-robin order -----------------------------------------------------

def test_rotated_order_starts_after_self():
    names = [f"receiver{i}" for i in range(9)]
    assert P.rotated_order(names, 3) == [f"receiver{i}" for i in (4, 5, 6, 7, 8, 0, 1, 2, 3)]
    assert P.rotated_order(names, 8) == names


def test_every_step_hits_each_receiver_once():
    # The reason for starting at i+1: with all nine senders sending one packet
    # per step (own receiver excluded), each of the first 8 steps maps the
    # senders one-to-one onto the receivers.
    names = [f"receiver{i}" for i in range(9)]
    orders = {i: [r for r in P.rotated_order(names, i) if r != f"receiver{i}"]
              for i in range(9)}
    for step in range(8):
        hit = sorted(orders[i][step] for i in range(9))
        assert len(set(hit)) == 9, f"step {step}: {hit}"


def test_app_data_waits_for_the_zero_second_timer(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver0"))
    assert sim.sent == []
    assert sim.timers == {P.SEND_TIMER: 0.0}


def test_later_app_data_does_not_reset_the_timer(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver0"))
    sim.t = 0.5   # a pending timer must not be pushed back by new messages
    s.on_app_data(msg(1, "receiver1"))
    assert sim.timers == {P.SEND_TIMER: 0.0}
    assert sim.sent == []


def test_a_destination_grouped_burst_goes_out_interleaved(tmp_path):
    # The harness offers a round grouped by destination: all of receiver0's
    # messages, then receiver1's, ... Sender 3 has nothing for itself.
    s, sim = make_sender(tmp_path)
    mid = 0
    for r in (0, 1, 2, 4, 5, 6, 7, 8):
        for _ in range(2):
            s.on_app_data(msg(mid, f"receiver{r}"))
            mid += 1
    fire(s, sim, P.SEND_TIMER)
    one_turn = [f"receiver{i}" for i in (4, 5, 6, 7, 8, 0, 1, 2)]
    assert sent_dsts(sim) == one_turn + one_turn


def test_each_queue_stays_fifo(tmp_path):
    s, sim = make_sender(tmp_path)
    for mid in range(3):
        s.on_app_data(msg(mid, "receiver5"))
    fire(s, sim, P.SEND_TIMER)
    assert [P.decode(d).msg_id for d, _ in sim.sent] == [0, 1, 2]


def test_rotation_continues_across_batches(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    fire(s, sim, P.SEND_TIMER)
    for mid, r in enumerate((4, 6), start=1):
        s.on_app_data(msg(mid, f"receiver{r}"))
    fire(s, sim, P.SEND_TIMER)
    assert sent_dsts(sim) == ["receiver4", "receiver6", "receiver4"]


def test_oversized_message_is_skipped_not_crashed(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4", size=P.MAX_PAYLOAD + 1))
    s.on_app_data(msg(1, "receiver4", size=P.MAX_PAYLOAD))
    fire(s, sim, P.SEND_TIMER)
    assert [P.decode(d).msg_id for d, _ in sim.sent] == [1]
    assert s.oversized == 1


# ---- receiver --------------------------------------------------------------

def test_receiver_delivers_once_and_acks_every_copy(tmp_path):
    m = msg(42, "receiver0", size=100)
    r, sim = make_receiver(tmp_path, [m])
    pkt = P.encode_data(42, m.payload, send_no=2)
    r.on_datagram(pkt, addr_of(3))
    r.on_datagram(pkt, addr_of(3))   # a resend: the first ACK may have been lost
    assert r.delivered(42)
    assert r.duplicates == 1
    assert sim.sent == [(P.encode_ack(42, 2), addr_of(3))] * 2   # echoes the copy


def test_receiver_does_not_ack_a_bad_payload(tmp_path):
    m = msg(7, "receiver0", size=100)
    r, sim = make_receiver(tmp_path, [m])
    r.on_datagram(P.encode_data(7, b"wrong bytes"), addr_of(3))
    assert not r.delivered(7)
    assert sim.sent == []   # no ACK, so the sender resends it


def test_receiver_ignores_junk(tmp_path):
    m = msg(1, "receiver0", size=100)
    r, sim = make_receiver(tmp_path, [m])
    r.on_datagram(b"\x00\x00", addr_of(3))
    r.on_datagram(P.encode_ack(1, 1), addr_of(3))
    assert r.malformed == 1
    assert not r.delivered(1)
    assert sim.sent == []
