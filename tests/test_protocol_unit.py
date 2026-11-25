"""Unit tests for protocol.py: packet format, round robin, pacing,
retransmission, RTT estimation and the receiver.

These drive MySender/MyReceiver directly with a fake simulator whose clock
only moves when a test runs its timers, so they run in milliseconds and need
no ns-3 build.
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
        self.sent = []        # (bytes, (ip, port))
        self.send_times = []  # sim.t of each send, same order as sent
        self.timers = {}      # name -> fire time

    def now(self):
        return self.t

    def send(self, data, ip, port):
        self.sent.append((data, (ip, port)))
        self.send_times.append(self.t)
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


def run_timers(endpoint, sim, until=float("inf"), only=None):
    """Fire due timers in time order, moving the clock, until none are left
    before `until`. `only` limits it to timers whose name starts with it."""
    while True:
        due = [(t, n) for n, t in sim.timers.items()
               if t <= until and (only is None or n.startswith(only))]
        if not due:
            return
        t, name = min(due)
        sim.t = t
        del sim.timers[name]
        endpoint.on_timer(name)


def drain(sender, sim):
    """Run the pacer until everything queued has been sent once."""
    run_timers(sender, sim, only=P.SEND_TIMER)


def msg(msg_id, dst, size=1400):
    return AppMessage(msg_id=msg_id, dst=dst, size=size, deadline_s=None, offered_s=0.0)


def sent_dsts(sim):
    by_addr = {addr_of(i): f"receiver{i}" for i in range(9)}
    return [by_addr[addr] for _, addr in sim.sent]


def sent_ids(sim):
    return [P.decode(d).msg_id for d, _ in sim.sent]


def gap_s(packet_len):
    return 8 * (packet_len + P.WIRE_HEADER_BYTES) / P.PACING_RATE_BPS


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
    drain(s, sim)
    one_turn = [f"receiver{i}" for i in (4, 5, 6, 7, 8, 0, 1, 2)]
    assert sent_dsts(sim) == one_turn + one_turn


def test_each_queue_stays_fifo(tmp_path):
    s, sim = make_sender(tmp_path)
    for mid in range(3):
        s.on_app_data(msg(mid, "receiver5"))
    drain(s, sim)
    assert sent_ids(sim) == [0, 1, 2]


def test_rotation_continues_across_batches(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    for mid, r in enumerate((4, 6), start=1):
        s.on_app_data(msg(mid, f"receiver{r}"))
    drain(s, sim)
    assert sent_dsts(sim) == ["receiver4", "receiver6", "receiver4"]


def test_oversized_message_is_skipped_not_crashed(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4", size=P.MAX_PAYLOAD + 1))
    s.on_app_data(msg(1, "receiver4", size=P.MAX_PAYLOAD))
    drain(s, sim)
    assert sent_ids(sim) == [1]
    assert s.oversized == 1


# ---- pacing ----------------------------------------------------------------

def test_pacing_rate_fits_the_rack_uplink():
    # 3 senders per rack, 6 of 8 destinations off-rack, 150 Mbps uplink.
    assert 3 * (6 / 8) * P.PACING_RATE_BPS <= 150e6


def test_sends_are_spaced_by_the_pacing_gap(tmp_path):
    s, sim = make_sender(tmp_path)
    for mid in range(4):
        s.on_app_data(msg(mid, f"receiver{mid}"))
    drain(s, sim)
    gaps = [b - a for a, b in zip(sim.send_times, sim.send_times[1:])]
    assert gaps == pytest.approx([gap_s(6 + 1400)] * 3)


def test_new_data_waits_out_the_current_gap(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    run_timers(s, sim, until=0.0, only=P.SEND_TIMER)   # first send at t=0
    assert sim.send_times == [0.0]
    sim.t = gap_s(1406) / 2                             # half a gap later...
    s.on_app_data(msg(1, "receiver5"))                  # ...new data arrives
    drain(s, sim)
    assert sim.send_times[1] == pytest.approx(gap_s(1406))


def test_data_after_an_idle_gap_goes_out_at_once(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)                                       # sent, then idle
    sim.t = 3 * gap_s(1406)
    s.on_app_data(msg(1, "receiver5"))
    drain(s, sim)
    assert sim.send_times[1] == pytest.approx(3 * gap_s(1406))


# ---- reliability -----------------------------------------------------------

def ack(sender, sim, msg_id, at, send_no=1, dst=0):
    sim.t = at
    sender.on_datagram(P.encode_ack(msg_id, send_no), addr_of(dst))


def test_every_send_arms_a_timeout_and_its_ack_clears_it(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    assert sim.timers == {"rto:0": P.RTO_INITIAL_S}
    ack(s, sim, 0, at=0.1)
    assert s.outstanding == {}
    assert sim.timers == {}
    assert s.rtt["receiver4"].srtt == pytest.approx(0.1)


def test_a_timeout_resends_the_same_bytes(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    run_timers(s, sim, until=P.RTO_INITIAL_S)
    first, second = (P.decode(d) for d, _ in sim.sent)
    assert (first.msg_id, first.payload) == (second.msg_id, second.payload)
    assert (first.send_no, second.send_no) == (1, 2)
    assert sim.sent[0][1] == sim.sent[1][1]
    assert sim.send_times[1] == pytest.approx(P.RTO_INITIAL_S)
    assert s.resent == 1


def test_resends_go_before_new_data(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    sim.t = P.RTO_INITIAL_S
    s.on_app_data(msg(1, "receiver5"))   # new data and a timeout at once
    s.on_timer("rto:0")
    del sim.timers["rto:0"]
    drain(s, sim)
    assert sent_ids(sim) == [0, 0, 1]


def test_repeated_timeouts_back_off_up_to_the_cap(tmp_path):
    # A message on a live path (its destination is not down): the first resend
    # waits one plain RTO (likely a single random loss); doubling starts once
    # the same message has timed out twice, and stops at RTO_MAX_S.
    s, _ = make_sender(tmp_path)
    rto = P.RTO_INITIAL_S
    waits = [s._timeout_for(P.Outstanding(0, "receiver4", b"", [0.0] * sends))
             for sends in range(1, 12)]
    assert waits[:4] == pytest.approx([rto, rto, 2 * rto, 4 * rto])
    assert max(waits) == waits[-1] == P.RTO_MAX_S


def test_a_lone_probe_backs_off_only_to_the_probe_cap(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    run_timers(s, sim, until=30.0)   # never ACKed: a blacked-out link
    waits = [b - a for a, b in zip(sim.send_times, sim.send_times[1:])]
    rto = P.RTO_INITIAL_S
    # Its third timeout marks receiver4 down and makes it the probe.
    assert waits[:3] == pytest.approx([rto, rto, 2 * rto])
    assert waits[3:] == pytest.approx([P.PROBE_MAX_S] * len(waits[3:]))
    assert len(waits) > 100   # it keeps probing every PROBE_MAX_S


def test_ack_for_a_resent_copy_is_timed_from_that_copy(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    run_timers(s, sim, until=P.RTO_INITIAL_S)   # resent once
    ack(s, sim, 0, at=P.RTO_INITIAL_S + 0.1, send_no=2)
    assert s.outstanding == {}
    assert s.rtt["receiver4"].srtt == pytest.approx(0.1)


def test_late_ack_for_the_first_copy_teaches_the_longer_rtt(tmp_path):
    # The path's RTT grew past the RTO: the first copy was resent, but its own
    # ACK still arrives. It is timed from the first send, so the estimate
    # learns the new RTT instead of staying frozen (Karn's rule would drop it).
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    run_timers(s, sim, until=P.RTO_INITIAL_S)
    ack(s, sim, 0, at=0.45, send_no=1)
    assert s.rtt["receiver4"].srtt == pytest.approx(0.45)


def test_estimator_follows_an_rtt_increase_to_the_end(tmp_path):
    # Every message's RTT jumps from 100 ms to 150 ms, above the RTO. Each one
    # times out and is resent, but the ACKs answer the first copies, so the
    # RTO climbs above 150 ms and the resends stop.
    s, sim = make_sender(tmp_path, receivers=[4])
    for _ in range(50):
        s.rtt["receiver4"].update(0.100)
    resent_before = []
    for mid in range(40):
        start = sim.t
        s.on_app_data(msg(mid, "receiver4"))
        run_timers(s, sim, until=start + 0.15)
        ack(s, sim, mid, at=start + 0.15, send_no=1)
        run_timers(s, sim, only=P.SEND_TIMER)
        resent_before.append(s.resent)
    assert s.rtt["receiver4"].rto() > 0.150
    assert resent_before[-1] == resent_before[-10]   # no resends at the end


def test_ack_while_waiting_to_resend_cancels_the_resend(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    sim.t = P.RTO_INITIAL_S
    del sim.timers["rto:0"]
    s.on_timer("rto:0")                     # queued for resend...
    ack(s, sim, 0, at=P.RTO_INITIAL_S)      # ...but the late ACK lands first
    drain(s, sim)
    assert sent_ids(sim) == [0]


def test_stray_and_repeated_acks_are_ignored(tmp_path):
    s, sim = make_sender(tmp_path)
    s.on_app_data(msg(0, "receiver4"))
    drain(s, sim)
    ack(s, sim, 0, at=0.1)
    ack(s, sim, 0, at=0.2)      # ACK for a duplicate copy
    ack(s, sim, 99, at=0.2)     # never sent
    s.on_datagram(b"\x01", addr_of(0))
    assert s.acked == 1


# ---- down destinations -----------------------------------------------------

def send_and_lose(sender, sim, ids, dst):
    """Offer `ids` to dst and let every one of them time out once."""
    for mid in ids:
        sender.on_app_data(msg(mid, dst))
    drain(sender, sim)
    run_timers(sender, sim, until=sim.t + P.RTO_INITIAL_S)


def test_isolated_timeouts_do_not_mark_a_destination_down(tmp_path):
    s, sim = make_sender(tmp_path)
    for mid in range(5):   # each loss is followed by an ACK from the same receiver
        send_and_lose(s, sim, [mid], "receiver4")
        ack(s, sim, mid, at=sim.t + 0.1, send_no=2)
    assert "receiver4" not in s.down


def test_timeouts_in_a_row_mark_a_destination_down(tmp_path):
    s, sim = make_sender(tmp_path)
    send_and_lose(s, sim, range(P.DOWN_AFTER_TIMEOUTS), "receiver4")
    assert s.down == {"receiver4"}


def test_a_down_destination_is_probed_by_one_message(tmp_path):
    s, sim = make_sender(tmp_path)
    send_and_lose(s, sim, range(6), "receiver4")      # 6 sends, 6 timeouts
    run_timers(s, sim, until=sim.t + 20.0)           # it stays dead
    resent_ids = sent_ids(sim)[6:]
    assert len(set(resent_ids)) <= P.DOWN_AFTER_TIMEOUTS   # resent before it went down
    probe = s.probe["receiver4"]
    assert resent_ids[-5:] == [probe] * 5             # afterwards only the probe
    assert len(s.parked["receiver4"]) >= 6 - P.DOWN_AFTER_TIMEOUTS


def test_a_down_destination_gets_no_new_data_but_others_do(tmp_path):
    s, sim = make_sender(tmp_path)
    send_and_lose(s, sim, range(P.DOWN_AFTER_TIMEOUTS), "receiver4")
    before = len(sim.sent)
    s.on_app_data(msg(100, "receiver4"))
    s.on_app_data(msg(101, "receiver5"))
    drain(s, sim)
    new = [i for i in sent_ids(sim)[before:] if i >= 100]
    assert new == [101]


def test_an_ack_brings_a_down_destination_back(tmp_path):
    s, sim = make_sender(tmp_path)
    send_and_lose(s, sim, range(6), "receiver4")
    s.on_app_data(msg(100, "receiver4"))              # held while down
    probe = s.probe["receiver4"]
    parked = list(s.parked["receiver4"])
    assert parked                                     # held, not resent
    ack(s, sim, probe, at=sim.t + 0.1, send_no=2)     # the server answers
    assert s.down == set()
    before = len(sim.sent)
    drain(s, sim)
    # Parked messages go out at once, ahead of the held new data. Messages
    # resent before it went down are still in flight on their own timers.
    assert sent_ids(sim)[before:] == parked + [100]


# ---- RTT estimator ---------------------------------------------------------

def test_rto_starts_at_the_initial_value():
    assert P.RttEstimator().rto() == P.RTO_INITIAL_S


def test_initial_rto_clears_the_base_rtt_but_not_by_much():
    # Above the 100 ms cross-rack RTT plus the queueing margin, so the first
    # round sees no needless resends; under two RTTs, so a first-round loss is
    # not penalised twice as hard as a later one.
    cross_rack_rtt = 0.100
    assert cross_rack_rtt + P.RTO_MARGIN_S < P.RTO_INITIAL_S < 2 * cross_rack_rtt


def test_first_sample_follows_rfc_6298():
    est = P.RttEstimator()
    est.update(0.1)
    assert (est.srtt, est.rttvar) == (0.1, 0.05)
    assert est.rto() == pytest.approx(0.1 + 4 * 0.05)


def test_steady_rtt_settles_at_srtt_plus_the_margin():
    est = P.RttEstimator()
    for _ in range(100):
        est.update(0.1)
    assert est.rto() == pytest.approx(0.1 + P.RTO_MARGIN_S)


def test_rto_is_clamped():
    est = P.RttEstimator()
    est.update(10.0)
    assert est.rto() == P.RTO_MAX_S
    est = P.RttEstimator()
    for _ in range(100):
        est.update(0.001)
    assert est.rto() == P.RTO_MIN_S


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
