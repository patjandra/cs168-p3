"""Data center backbone protocol: pacing, round robin, ACKs and timeouts.

Sender
    One FIFO queue per destination, served round robin, one packet per
    destination per turn. Sender i starts at receiver i+1 (see rotated_order).
    Packets leave at a fixed pacing rate, never in back-to-back bursts.
    Every DATA packet waits in an outstanding table until its ACK arrives. If
    its RTO timer fires first, it is resent ahead of new data. The first
    resend waits one plain RTO; from the second timeout of the same message
    on, each wait doubles (capped). Each destination has its own RTT estimate.
    A destination that keeps timing out is treated as down: its new data and
    timed-out messages are held, and one message probes it on a backed-off
    timer until an ACK shows it is back.

Receiver
    Delivers each message once and ACKs every copy it holds, so a lost ACK is
    repaired by the next resend.

Packet format (network byte order):

    DATA = [type: 1 byte = 0][send number: 1 byte][message ID: 4 bytes][payload]
    ACK  = [type: 1 byte = 1][send number: 1 byte][message ID: 4 bytes]

The send number counts transmissions of one message (1, 2, ...), and an ACK
echoes the number of the copy it answers. The sender can then time every
ACK against the exact copy it answers, resends included. Without it, Karn's
rule must discard every resent message's ACK, and once the path's RTT grows
past the RTO, every message is resent and no sample is ever valid again.

Where things live, for Stage 3 tuning:

    pacing rate              PACING_RATE_BPS; the gap is set in _transmit
    send order               _next_packet: resends first, then _next_message
    RTO                      RttEstimator, RTO_* constants, _timeout_for
    down destinations        DOWN_AFTER_TIMEOUTS, _on_timeout, _heard_from
    dynamic pacing           would adjust self.rate_bps in on_datagram and
                             _on_timeout

Run it with `make run SCENARIO=datacenter`.
"""

from __future__ import annotations

import re
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

from transportlab.api import Receiver, Sender

# The scenario your team chose: "leo", "datacenter" or "sensor". Gradescope
# grades your protocol on this scenario, and rejects it while this is empty.
SCENARIO = "datacenter"


# ---------------------------------------------------------------------------
# Packet format (shared by sender and receiver)
# ---------------------------------------------------------------------------

TYPE_DATA = 0
TYPE_ACK = 1

HEADER = struct.Struct("!BBI")           # type, send number, message ID -> 6 bytes
MAX_DATAGRAM = 1472                       # send_datagram's limit
MAX_PAYLOAD = MAX_DATAGRAM - HEADER.size  # 1466
MAX_SEND_NO = 255                         # the send number saturates here

SEND_TIMER = "send"   # fires the round-robin pump
LOG_SENDS = 20        # log only the first few sends; logs are capped at 5000 lines


def encode_data(msg_id: int, payload: bytes, send_no: int = 1) -> bytes:
    """DATA for one copy of a message; send_no 1 is its first transmission."""
    return HEADER.pack(TYPE_DATA, min(send_no, MAX_SEND_NO), msg_id) + payload


def encode_ack(msg_id: int, send_no: int) -> bytes:
    """ACK one copy of a message; send_no is echoed from that copy's DATA."""
    return HEADER.pack(TYPE_ACK, send_no, msg_id)


class Packet(NamedTuple):
    kind: int
    send_no: int     # which transmission of the message (1 = first)
    msg_id: int
    payload: bytes


def decode(data: bytes) -> Packet | None:
    """Return the packet, or None for one we don't understand."""
    if len(data) < HEADER.size:
        return None
    kind, send_no, msg_id = HEADER.unpack_from(data)
    if kind == TYPE_DATA:
        return Packet(kind, send_no, msg_id, bytes(data[HEADER.size:]))
    if kind == TYPE_ACK and len(data) == HEADER.size:
        return Packet(kind, send_no, msg_id, b"")
    return None


def host_index(name: str) -> int | None:
    """The trailing number of a host name: "sender3" -> 3, "receiver0" -> 0."""
    m = re.search(r"\d+$", name)
    return int(m.group()) if m else None


def rotated_order(receivers, me: int | None) -> list[str]:
    """Receivers in round-robin order, starting after receiver `me` and wrapping.

    Why start at me+1: if every sender sends one packet per step, sender i
    sends to receiver (i+1+t) mod 9 at step t. For each t that maps the nine
    senders one-to-one onto the nine receivers, so every receiver gets exactly
    one packet per step. Each sender's own receiver comes last (step 8), so
    all senders skip it at the same step and stay in line.

    Receivers without a number go last, in name order.
    """
    numbered = sorted((r for r in receivers if host_index(r) is not None), key=host_index)
    others = sorted(r for r in receivers if host_index(r) is None)
    if me is not None:
        start = next((i for i, r in enumerate(numbered) if host_index(r) > me), 0)
        numbered = numbered[start:] + numbered[:start]
    return numbered + others


# ---------------------------------------------------------------------------
# Sender constants
# ---------------------------------------------------------------------------

WIRE_HEADER_BYTES = 28   # IPv4 + UDP, counted on the wire for every datagram

# ---------------------------------------------------------------------------
# Pacing
#
# The bottleneck is each rack's 150 Mbps link to the spine. Three senders
# share it, and 6 of each sender's 8 destinations are in other racks, so at
# sending rate r the uplink carries 3 * (6/8) * r. That fits while
# r <= 150 / 2.25 = 66.7 Mbps. The spine-to-rack links are symmetric, and a
# receiver's 100 Mbps link carries 8 * (r/8) = r, so 66.7 Mbps is the binding
# share. Pacing under it leaves the 64-packet switch queues room for the
# moments when all three senders in a rack aim off-rack at once, and for a
# link that slows down.
#
# Tuned in testing (tools/run_suite.py, seeds 1-4): 90 Mbps overflows the
# uplinks on 24-packet bursts (3.0x slowdown); 60-75 Mbps finish them in
# 1.01-1.05x but lose packets when an uplink drops to 75 Mbps (2.0x). 55 Mbps
# (82% of the share) costs 1.09x on big bursts and loses nothing on the slow
# uplink (1.02x).
# ---------------------------------------------------------------------------

PACING_RATE_BPS = 55e6

# ---------------------------------------------------------------------------
# Retransmission timeout (RFC 6298 smoothing, one estimator per destination)
#
# Base RTTs are 80 ms inside a rack and 100 ms across racks. RTTVAR shrinks
# below 1 ms after a few steady rounds, so 4 x RTTVAR alone would put the RTO
# right on top of the RTT. A full 64-packet queue adds up to about 7 ms per
# hop, and a cross-rack packet crosses three queued hops, so the RTO never
# sits less than RTO_MARGIN_S (20 ms) above the smoothed RTT. A sudden queue
# then delays an ACK rather than triggering a resend.
#
# Before a path's first sample, RTO_INITIAL_S is 1.5 cross-rack RTTs: above
# the 80-100 ms base RTT plus any queueing, but low enough that a loss in
# the first round does not wait twice as long as one later on. RFC 6298's
# 1 s (and our earlier 300 ms) is for networks whose RTT is unknown; this
# one's is fixed by the topology. At 300 ms the first round of a lossy run
# took 8.0x the ideal against about 3.0x for the rest; 150 ms cut 0.5-1%
# uplink loss from 3.6x to 3.3x mean, with no extra resends on clean runs.
#
# Backoff: one timeout is almost always a single random loss, and waiting
# longer for that resend only stretches the round. Two timeouts in a row on
# the same message look like a dead path (a failed server, a blackout), so
# from then on each timeout doubles, up to RTO_MAX_S, and the dead path is
# probed less and less often instead of flooded. On 2% uplink loss this cut
# mean step slowdown by about 25% against doubling from the first timeout.
#
# Down destinations: per-message backoff alone still probes a dead server
# once per stuck message. After DOWN_AFTER_TIMEOUTS timeouts in a row to one
# destination, with no ACK from it in between, the destination is marked
# down. Random loss rarely does that, because ACKs for the messages around a
# lost one keep resetting the count. While down, the destination gets no new
# data, its timed-out messages are parked, and a single probe message keeps
# backing off, but only up to PROBE_MAX_S: one probe is not a flood, and a
# short interval finds the server soon after it returns. The first ACK from
# it brings everything back. Measured on a server down for 1 s, by probe
# cap: 2 s -> 1.80x slowdown, 0.5 s -> 1.53x, 0.25 s -> 1.10x (overhead
# 1.13). A permanently dead server then costs overhead 1.23, against 2.61
# when every stuck message probed on its own.
# ---------------------------------------------------------------------------

RTT_ALPHA = 1 / 8
RTT_BETA = 1 / 4
RTT_K = 4
RTO_INITIAL_S = 0.150
RTO_MARGIN_S = 0.020
BACKOFF_AFTER_SENDS = 2   # sends at the base RTO before doubling starts
RTO_MIN_S = 0.050
RTO_MAX_S = 2.0
DOWN_AFTER_TIMEOUTS = 3
PROBE_MAX_S = 0.25

RTO_PREFIX = "rto:"   # "rto:<msg_id>" fires that message's timeout
LOG_TIMEOUTS = 50     # log only the first few timeouts

# ---------------------------------------------------------------------------
# Loss detection from ACKs (RACK, RFC 8985)
#
# A lost packet otherwise waits out its whole RTO (about 120 ms) before it is
# resent. But packets to one destination share one FIFO path, so once a later
# packet to that destination is ACKed, an earlier unACKed one is lost unless
# the network reordered them. A packet sent at least the reordering window
# before an ACKed one is therefore resent straight away, about one RTT after
# it left instead of one RTO. Packets to a destination leave about 1.7 ms
# apart during a round, so the loss shows up within a packet or two.
#
# Only per-packet delay jitter reorders packets here. The send number echoed
# in each ACK shows when an early resend was needless (the first copy is the
# one ACKed), and that destination's window then doubles, up to
# REO_WND_MAX_S. A loss with no later packet behind it (the last one to a
# destination in a round) still waits for its RTO.
# ---------------------------------------------------------------------------

REO_WND_MIN_S = 0.002
REO_WND_MAX_S = 0.020


class RttEstimator:
    """Smoothed RTT and its variation for one path (RFC 6298)."""

    def __init__(self):
        self.srtt: float | None = None
        self.rttvar: float | None = None

    def update(self, sample: float):
        if self.srtt is None:
            self.srtt, self.rttvar = sample, sample / 2
        else:
            self.rttvar = (1 - RTT_BETA) * self.rttvar + RTT_BETA * abs(self.srtt - sample)
            self.srtt = (1 - RTT_ALPHA) * self.srtt + RTT_ALPHA * sample

    def rto(self) -> float:
        if self.srtt is None:
            return RTO_INITIAL_S
        rto = self.srtt + max(RTO_MARGIN_S, RTT_K * self.rttvar)
        return min(max(rto, RTO_MIN_S), RTO_MAX_S)


@dataclass(slots=True)
class Outstanding:
    """A message sent at least once and not yet ACKed."""

    msg_id: int
    dst: str
    payload: bytes     # read once: msg.payload is rebuilt from SHA-256 on every read
    send_times: list[float] = field(default_factory=list)   # [n-1] = send n
    awaiting_resend: bool = False   # queued or parked for a resend, timer stopped
    early_resend_no: int = 0        # send number of a resend that RACK triggered

    @property
    def sends(self) -> int:
        return len(self.send_times)

    def sent_at(self, send_no: int) -> float | None:
        """When copy `send_no` left, or None if an echoed number can't say."""
        if 1 <= send_no < MAX_SEND_NO and send_no <= len(self.send_times):
            return self.send_times[send_no - 1]
        return None


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

class MySender(Sender):
    def on_start(self):
        self.me = host_index(self.name)
        receivers = [p for p in self.peers if p.startswith("receiver")] or list(self.peers)
        self.order = rotated_order(receivers, self.me)
        self.queues = {dst: deque() for dst in self.order}
        self.rr = 0   # position in self.order of the next destination to try

        self.rate_bps = PACING_RATE_BPS
        self.next_send_at = 0.0          # pacing: no send before this time
        self.outstanding: dict[int, Outstanding] = {}   # msg_id -> unACKed packet
        self.resend: deque[int] = deque()               # timed-out msg_ids
        self.rtt = {dst: RttEstimator() for dst in self.order}

        # Down detection (see DOWN_AFTER_TIMEOUTS)
        self.timeouts_in_row = {dst: 0 for dst in self.order}
        self.down: set[str] = set()
        self.probe: dict[str, int] = {}                  # dst -> probing msg_id
        self.parked = {dst: deque() for dst in self.order}   # held msg_ids

        # Loss detection from ACKs (see REO_WND_MIN_S)
        self.inflight = {dst: deque() for dst in self.order}   # (sent_at, msg_id)
        self.reo_wnd = {dst: REO_WND_MIN_S for dst in self.order}

        self.data_sent = 0
        self.resent = 0
        self.early_resends = 0
        self.needless_resends = 0
        self.timeouts = 0
        self.acked = 0
        self.oversized = 0
        self.log("order", " ".join(self.order))

    def on_app_data(self, msg):
        if msg.size > MAX_PAYLOAD:
            # No fragmentation yet: skip it rather than crash the trial.
            self.oversized += 1
            self.log("oversized: skipping msg", msg.msg_id, "of", msg.size, "bytes")
            return
        self.queues[msg.dst].append(msg)
        self._wake()

    def on_timer(self, name):
        if name == SEND_TIMER:
            self._pump()
        elif name.startswith(RTO_PREFIX):
            self._on_timeout(int(name[len(RTO_PREFIX):]))

    def on_datagram(self, data, addr):
        pkt = decode(data)
        if pkt is None or pkt.kind != TYPE_ACK:
            return
        entry = self.outstanding.pop(pkt.msg_id, None)
        if entry is None:
            return   # an ACK for another copy, after the first ACK
        self.cancel_timer(RTO_PREFIX + str(pkt.msg_id))
        self.acked += 1
        sent_at = entry.sent_at(pkt.send_no)   # the copy this ACK answers
        if sent_at is not None:
            self.rtt[entry.dst].update(self.now() - sent_at)
        if 0 < pkt.send_no < entry.early_resend_no:
            self._early_resend_was_needless(entry.dst)
        self._heard_from(entry.dst)
        if sent_at is not None:
            self._detect_losses(entry.dst, sent_at)

    def on_stop(self):
        self.log("data_sent", self.data_sent, "resent", self.resent,
                 "early", self.early_resends, "needless", self.needless_resends,
                 "acked", self.acked, "unacked", len(self.outstanding),
                 "oversized", self.oversized)
        for dst in self.order:
            est = self.rtt[dst]
            if est.srtt is not None:
                self.log("rtt", dst, f"srtt={est.srtt * 1e3:.1f}ms",
                         f"rttvar={est.rttvar * 1e3:.1f}ms", f"rto={est.rto() * 1e3:.1f}ms")

    # ---- pacing -------------------------------------------------------------

    def _wake(self):
        """Make sure the pacer will run, but not before the pacing gap is up.

        The harness offers a whole round in one burst, grouped by destination.
        Even with no gap to wait out, the pacer runs from a 0 s timer, which
        fires after the burst is queued, so round robin sees every destination.
        """
        if not self.timer_pending(SEND_TIMER):
            self.set_timer(SEND_TIMER, max(0.0, self.next_send_at - self.now()))

    def _pump(self):
        """Send one packet, then wait one pacing gap before the next."""
        wait = self.next_send_at - self.now()
        if wait > 1e-9:   # timers round to whole nanoseconds
            self.set_timer(SEND_TIMER, wait)
            return
        entry = self._next_packet()
        if entry is None:
            return   # idle until new data or a timeout calls _wake
        self._transmit(entry)
        self.set_timer(SEND_TIMER, self.next_send_at - self.now())

    def _next_packet(self) -> Outstanding | None:
        """Resends first, oldest first; then new data, round robin."""
        while self.resend:
            entry = self.outstanding.get(self.resend.popleft())
            if entry is not None:   # None: ACKed while it waited
                return entry
        picked = self._next_message()
        if picked is None:
            return None
        dst, msg = picked
        entry = Outstanding(msg.msg_id, dst, msg.payload)
        self.outstanding[msg.msg_id] = entry
        return entry

    def _next_message(self):
        """Round robin: (dst, msg) from the next non-empty queue, or None.

        Destinations marked down are skipped until they answer a probe.
        """
        n = len(self.order)
        for step in range(n):
            i = (self.rr + step) % n
            q = self.queues[self.order[i]]
            if q and self.order[i] not in self.down:
                self.rr = (i + 1) % n
                return self.order[i], q.popleft()
        return None

    # ---- reliability --------------------------------------------------------

    def _transmit(self, entry: Outstanding):
        now = self.now()
        packet = encode_data(entry.msg_id, entry.payload, send_no=entry.sends + 1)
        self.send_datagram(packet, to=entry.dst)
        entry.send_times.append(now)
        entry.awaiting_resend = False
        self.inflight[entry.dst].append((now, entry.msg_id))
        self.data_sent += 1
        if entry.sends > 1:
            self.resent += 1
        self.set_timer(RTO_PREFIX + str(entry.msg_id), self._timeout_for(entry))
        wire_bits = 8 * (len(packet) + WIRE_HEADER_BYTES)
        self.next_send_at = now + wire_bits / self.rate_bps
        if self.data_sent <= LOG_SENDS:
            self.log("send", entry.dst, entry.msg_id)

    def _timeout_for(self, entry: Outstanding) -> float:
        """The path's RTO; doubled per send past BACKOFF_AFTER_SENDS, capped.

        A down destination's single probe is capped lower, at PROBE_MAX_S: one
        packet per interval costs almost nothing, and a short interval finds
        the server soon after it comes back.
        """
        doublings = max(0, entry.sends - BACKOFF_AFTER_SENDS)
        cap = RTO_MAX_S
        if entry.dst in self.down and self.probe.get(entry.dst) == entry.msg_id:
            cap = PROBE_MAX_S
        return min(self.rtt[entry.dst].rto() * 2 ** doublings, cap)

    def _on_timeout(self, msg_id: int):
        entry = self.outstanding.get(msg_id)
        if entry is None:
            return   # ACKed in the same instant the timer fired
        dst = entry.dst
        self.timeouts += 1
        if self.timeouts <= LOG_TIMEOUTS:
            self.log("timeout", dst, msg_id, "sends", entry.sends)
        self.timeouts_in_row[dst] += 1
        if dst not in self.down and self.timeouts_in_row[dst] >= DOWN_AFTER_TIMEOUTS:
            self.down.add(dst)
            self.log("down", dst, "after", self.timeouts_in_row[dst], "timeouts")
        entry.awaiting_resend = True
        if dst in self.down:
            probe = self.probe.get(dst)
            if probe is not None and probe != msg_id and probe in self.outstanding:
                self.parked[dst].append(msg_id)   # one probe at a time
                return
            self.probe[dst] = msg_id
        self.resend.append(msg_id)
        self._wake()

    def _detect_losses(self, dst: str, acked_sent_at: float):
        """Resend packets to dst sent a reordering window before an ACKed one."""
        q = self.inflight[dst]
        found = False
        while q and q[0][0] + self.reo_wnd[dst] <= acked_sent_at:
            sent_at, msg_id = q.popleft()
            entry = self.outstanding.get(msg_id)
            if entry is None or entry.awaiting_resend or entry.send_times[-1] != sent_at:
                continue   # ACKed, already queued, or resent since this copy
            self.cancel_timer(RTO_PREFIX + str(msg_id))
            entry.awaiting_resend = True
            entry.early_resend_no = entry.sends + 1
            self.early_resends += 1
            self.resend.append(msg_id)
            found = True
        if found:
            self._wake()

    def _early_resend_was_needless(self, dst: str):
        """An earlier copy got through after all: the path reorders more."""
        self.needless_resends += 1
        self.reo_wnd[dst] = min(2 * self.reo_wnd[dst], REO_WND_MAX_S)

    def _heard_from(self, dst: str):
        """An ACK came back from dst: it is reachable, so release what was held."""
        self.timeouts_in_row[dst] = 0
        if dst not in self.down:
            return
        self.down.discard(dst)
        self.probe.pop(dst, None)
        self.log("up", dst, "releasing", len(self.parked[dst]), "parked")
        self.resend.extend(self.parked[dst])
        self.parked[dst].clear()
        self._wake()


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class MyReceiver(Receiver):
    def on_start(self):
        self.malformed = 0
        self.duplicates = 0

    def on_datagram(self, data, addr):
        pkt = decode(data)
        if pkt is None:
            self.malformed += 1
            return
        if pkt.kind != TYPE_DATA:
            return
        if self.delivered(pkt.msg_id):
            self.duplicates += 1   # a resend whose first copy got through
        else:
            self.deliver(pkt.msg_id, pkt.payload)
        # ACK only what we hold, so a bad copy is resent. A duplicate is ACKed
        # again because it means our first ACK may have been lost. The echoed
        # send number lets the sender time this exact copy.
        if self.delivered(pkt.msg_id):
            self.send_datagram(encode_ack(pkt.msg_id, pkt.send_no), to=addr)

    def on_stop(self):
        self.log("malformed", self.malformed, "duplicates", self.duplicates)


SENDER, RECEIVER = MySender, MyReceiver
